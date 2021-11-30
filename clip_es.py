
#%%

#### 
#### Optimizing a collage of small images using Genetic Algorithm
#### based on the output of OpenAI CLIP model.
####
#### "Arrange given small images so that it looks like xxxx 
#### as much as possible"
####

from glob import glob
import random
from PIL import Image, ImageDraw
import IPython.display as ipd
import math
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.makedirs("./output", exist_ok=True)

import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

#%%

imagepaths = glob("./images/fukuwarai/*.png") # small images that consist of the collage
NUM_IMAGES = len(imagepaths)
print("# of images: ", NUM_IMAGES)

CANVAS_SIZE = 900 # the size of the collage

NUM_IMAGES_IN_GENE = 12 # how many small images in one collage
DEFAULT_IMG_WIDTH = 225 # how big these small images should be in pixel

GENE_LENGTH = 5 # number of genes for one small image

### This is the target! 
TARGET_TEXT = "a illustration of a face of a boy"

#%%
### Initialize CLIP model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# %%

def draw_gene(genes, canvas):
    
    num_genes = len(genes) // GENE_LENGTH

    for i in range(num_genes):
        gene = genes[i*GENE_LENGTH:(i+1)*GENE_LENGTH]
        img_index = int(abs(gene[0]) * NUM_IMAGES) % NUM_IMAGES
        x = int(gene[1]%1.0 * CANVAS_SIZE)
        y = int(gene[2]%1.0 * CANVAS_SIZE)
        w = pow((gene[3] + 0.5) * DEFAULT_IMG_WIDTH, 2)
        rot = gene[4] * 360.0

        impath = imagepaths[img_index]
        im= Image.open(impath)
        org_w, org_h = im.size
        h = int(org_h * (DEFAULT_IMG_WIDTH / org_w))
        im.thumbnail((DEFAULT_IMG_WIDTH, h))
        im = im.rotate(rot, expand=True)

        canvas.alpha_composite(im, (x, y))
    return canvas

class Individual(object):
    
    def __init__(self, numbers=None, error_weight=0.0):
        if numbers is None:
            self.numbers = np.random.randn(GENE_LENGTH * NUM_IMAGES_IN_GENE) + 0.5 # randomize initial gene
            # print(self.numbers[:10])
        else: 
            self.numbers = numbers # assign gene          
            self.numbers += error_weight * np.random.randn(*self.numbers.shape)           
        self.reset()

    def reset(self):
        self.generated_img = None
        self.fitness_value = -1
        self.canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))

    def generate_image(self):
        self.generated_img = draw_gene(self.numbers, self.canvas)

    def save_image(self, path):
        if self.generated_img is None:
            self.generate_image()
        self.generated_img.save(path)

    def fitness(self):
        if self.fitness_value >= 0.0:
          return self.fitness_value
        if self.generated_img is None:
            self.generate_image()
        
        image = preprocess(self.generated_img).unsqueeze(0).to(device)
        text = clip.tokenize([TARGET_TEXT]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy()
            text_features = model.encode_text(text).cpu().numpy()
    
            sims = cosine_similarity(image_features, text_features)
        self.fitness_value = float(sims)
        return self.fitness_value
    
    def show(self):
        c = draw_gene(self.numbers, self.canvas)
        ipd.display(c)
# %%

class Population(object):

    def __init__(self, pop_size=100, elite=5, top_k=20, error_weight = 1, decay_rate = 0.95, min_error_weight = 0.01 ):
        """
            Args
                pop_size: size of population
        """
        self.pop_size = pop_size
        self.elite = elite
        self.top_k = top_k
        self.error_weight = error_weight
        self.decay_rate = decay_rate
        self.top_k = top_k
        self.min_error_weight = min_error_weight
        self.fitness_history = []
        self.best_gene_history = []
        self.parents = []
        self.done = False

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(Individual())
            
    def generate_images(self):
        # for indiv in tqdm(self.individuals):
        #      indiv.generate_image()
        Parallel(n_jobs=-1,prefer="threads")(delayed(indiv.generate_image)() for indiv in self.individuals)

    def get_best_individual(self):
        return list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=False)))[0]

    def save_best(self, dir, prefix, generation):
        best_indiv = self.get_best_individual()
        path = os.path.join(dir, "%0.4f_%s_%d.png" % (best_indiv.fitness(), prefix, generation))
        print("saved at: ", path)
        best_indiv.save_image(path)

    def grade(self, generation=None):
        """
            Grade the generation by getting the average fitness of its individuals
        """

        fitness_sum = 0
        for x in self.individuals:
            fitness_sum += x.fitness()

        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)
        
        best_indv = self.get_best_individual()   
        #self.image_history.append(best_indv.generated_img)
        self.best_gene_history.append(best_indv.numbers)
               
#         # Set Done flag if we hit target
#         if int(round(pop_fitness)) == 0:
#             self.done = True

        if generation is not None:
            if generation % 1 == 0:
                print("Episode",generation,"Population fitness:", pop_fitness)
                print("Weight:", self.error_weight)
                print("Best: ", best_indv.fitness())
                ipd.display(best_indv.generated_img)
            
    def select_parents_and_breed(self):
        """
            Select the fittest individuals to be the parents of next generation (lower fitness it better in this case)
            Also select a some random non-fittest individuals to help get us out of local maximums
        """
        # Sort individuals by fitness
        self.individuals = list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=False)))

        # select top_k and mean them
        best_pop = []
        for p in self.individuals[:self.top_k]:
            best_pop.append(p.numbers)
        mean_best_indiv_gene = np.mean(np.array(best_pop), axis=0)

        # generate the next gen
        next_gen = []
        for i in range(self.elite):
            next_gen.append(Individual(self.individuals[i].numbers))#, self.min_error_weight))
        for _ in range(self.pop_size - self.elite):
            next_gen.append(Individual(mean_best_indiv_gene, self.error_weight))
        self.error_weight = max(self.min_error_weight, self.error_weight * self.decay_rate)
        
        # replace the current population
        self.individuals = next_gen
    # def breed(self):
    #     """
    #         Crossover the parents to generate children and new generation of individuals
    #     """
    #     target_children_size = self.pop_size - len(self.parents)
        
    #     if len(self.parents) > 0:          
    #         while len(self.children) < target_children_size:
    #             father = random.choice(self.parents)
    #             mother = random.choice(self.parents)
    #             if father != mother:
    #                 child_numbers = [ random.choice(pixel_pair) for pixel_pair in zip(father.numbers, mother.numbers)]
    #                 child = Individual(child_numbers, mutate_prob=mutate_prob)
    #                 self.children.append(child)
    #         self.individuals = self.parents + self.children

#     def evolve(self):
#         # 1. Select fittest
#         self.select_parents_and_breed()
#         # 2. Create children and new generation
#         # self.breed()
#         # 3. Reset parents and children
# 
    def reset(self):
        for indiv in self.individuals:
            indiv.reset()

# %%

pop_size = 300

pop = Population(pop_size=pop_size, elite=10, top_k=30, error_weight=0.50, decay_rate=0.99)

SHOW_PLOT = True
GENERATIONS = 5000
for x in range(GENERATIONS):
    print("Generation #", x)
    print("generate images...")
    pop.generate_images()
    print("grade images and breed...")
    pop.grade(generation=x)
    pop.save_best("output", 'best_es', x)
    pop.select_parents_and_breed()

    if pop.done:
        print("Finished at generation:", x, ", Population fitness:", pop.fitness_history[-1])
        break


# %%
2.8 % 1.0
# %%
np.random.randn(*[2,3])
# %%
