
#%%

from glob import glob
import random
from PIL import Image
import IPython.display as ipd
import math
import numpy as np

imagepaths = glob("./images/*.png")
NUM_IMAGES = len(imagepaths)

CANVAS_SIZE = 1200
DEFAULT_IMG_WIDTH = 200

GENE_LENGTH = 5

# %%
def draw_gene(genes, canvas):

    for gene in genes:
        img_index = int(abs(gene[0]) * NUM_IMAGES) % NUM_IMAGES
        x = int(gene[1] * CANVAS_SIZE)
        y = int(gene[2] * CANVAS_SIZE)
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
# %%

import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# %%

# 
# image = preprocess(c).unsqueeze(0).to(device)
# text = clip.tokenize(["image of various fruits", "image of cars"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image).cpu().numpy()
#     text_features = model.encode_text(text).cpu().numpy()
    
#     sims = cosine_similarity(image_features, text_features)
#     print(sims)
# %%

TARGET_TEXT = "an image of a boy"

from joblib import Parallel, delayed

def draw_gene(genes, canvas):
    
    num_genes = len(genes) // GENE_LENGTH

    for i in range(num_genes):
        gene = genes[i*GENE_LENGTH:(i+1)*GENE_LENGTH]
        img_index = int(abs(gene[0]) * NUM_IMAGES) % NUM_IMAGES
        x = int(gene[1] * CANVAS_SIZE)
        y = int(gene[2] * CANVAS_SIZE)
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


NUM_IMAGES_IN_GENE = 40

class Individual(object):
    
    def __init__(self, numbers=None, mutate_prob=0.01):
        if numbers is None:
            self.numbers = np.random.rand(GENE_LENGTH * NUM_IMAGES_IN_GENE) # randomize initial gene
        else: 
            self.numbers = numbers # assign gene
            # Mutate
            if mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(self.numbers) - 1)
                self.numbers[mutate_index] = np.random.rand()                
        self.reset()

    def reset(self):
        self.generated_img = None
        self.fitness_value = -1
        self.canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE))

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
from tqdm import tqdm
import os

class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=0.2, random_retain=0.03, random_gen=0.05, elites=0.0):
        """
            Args
                pop_size: size of population
        """
        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.random_gen = random_gen
        self.elites = elites
        self.fitness_history = []
        self.image_history = []
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

    def save_best(self, dir, generation):
        best_indiv = self.get_best_individual()
        path = os.path.join(dir, "best_%d_%0.4f.png" % (generation, best_indiv.fitness()))
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

                print("Best: ", best_indv.fitness())
                ipd.display(best_indv.generated_img)
            
    def select_parents(self):
        """
            Select the fittest individuals to be the parents of next generation (lower fitness it better in this case)
            Also select a some random non-fittest individuals to help get us out of local maximums
        """
        # Sort individuals by fitness
        self.individuals = list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=False)))
        # Keep the fittest as parents for next gen
        retain_length = self.retain * len(self.individuals)
        for p in self.individuals[:int(retain_length)]:
            self.parents.append(Individual(p.numbers, mutate_prob=mutate_prob))

        # keep the best ones intact to the next generation
        elite_length = self.elites * len(self.individuals)
        self.children =  self.individuals[:int(elite_length)].copy()
        
        # Randomly select some from unfittest and add to parents array
        unfittest = self.individuals[int(retain_length):]
        unfittest = random.sample(unfittest, int(self.random_retain * len(self.individuals)))
        self.parents.extend(unfittest)

        # randomly generate new indivisuals
        nb_random_new = int(self.random_gen * len(self.individuals))
        for _ in range(nb_random_new):
          self.parents.append(Individual())
         
        print ("parents:", len(self.parents))
        print ("elite children:", len(self.children))
        
        
    def breed(self):
        """
            Crossover the parents to generate children and new generation of individuals
        """
        target_children_size = self.pop_size - len(self.parents)
        
        if len(self.parents) > 0:          
            while len(self.children) < target_children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                if father != mother:
                    child_numbers = [ random.choice(pixel_pair) for pixel_pair in zip(father.numbers, mother.numbers)]
                    child = Individual(child_numbers, mutate_prob=mutate_prob)
                    self.children.append(child)
            self.individuals = self.parents + self.children

    def evolve(self):
        # 1. Select fittest
        self.select_parents()
        # 2. Create children and new generation
        self.breed()
        # 3. Reset parents and children
        self.parents = []
        self.children = []

    def reset(self):
        for indiv in self.individuals:
            indiv.reset()

# %%

pop_size = 100
mutate_prob = 0.20
retain = 0.25
random_retain = 0.05
random_gen = 0.05
elites = 0.05

pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain, random_gen=random_gen, elites=elites)

SHOW_PLOT = True
GENERATIONS = 500
for x in range(GENERATIONS):
    print("generate images...")
    pop.generate_images()
    print("grade images and breed...")
    pop.grade(generation=x)
    pop.evolve()
    pop.save_best("output", x)
    pop.reset()

    if pop.done:
        print("Finished at generation:", x, ", Population fitness:", pop.fitness_history[-1])
        break
# %%

# a = pop.individuals[0]
# b = pop.individuals[70]
# print(a.numbers[:30])
# print(b.numbers[:30])
# c = [ random.choice(pixel_pair) for pixel_pair in zip(a.numbers, b.numbers)]
# print(c[:30])
# %%
