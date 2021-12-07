
#%%

#### 
#### Optimizing a collage of small images using PGPE
#### based on the output of OpenAI CLIP model.
####
#### "Arrange given small images so that it looks like xxxx 
#### as much as possible"
####

from glob import glob
from pgpelib import PGPE
from PIL import Image, ImageDraw, ImageFont
import IPython.display as ipd
import math
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os

import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES']='2'
os.makedirs("./output_text", exist_ok=True)

#%%

ALPHABETS = ["い","ろ","は","に","ほ", "へ", "と", "ち","り","ぬ","る","を","こ"]
#ALPHABETS = ["A","a", "B", "b", "C", "c", "D", "d", "E", "e", "F","f", "G", "g"]
ALPHABET_FONT = ImageFont.truetype("ipam.ttf", size=195)
ALPHABET_SIZE = 200 
ALPHABET_IMAGES = []
NUM_IMAGES = len(ALPHABETS)

for t in ALPHABETS:
    canvas = Image.new('RGBA', (ALPHABET_SIZE, ALPHABET_SIZE))

    textcolor = (0, 0, 0) 
    draw = ImageDraw.Draw(canvas) 
    draw.text((5, 5), t, font=ALPHABET_FONT, fill=textcolor)
    ipd.display(canvas)
    ALPHABET_IMAGES.append(canvas)
#%%

CANVAS_SIZE = 500 # the size of the collage canvas

NUM_IMAGES_IN_GENE = NUM_IMAGES  # how many small images in one collage
DEFAULT_IMG_WIDTH = ALPHABET_SIZE # how big these small images should be in pixel

GENE_LENGTH = 4 # number of genes for one small image

SOLUTION_LENGTH = GENE_LENGTH * NUM_IMAGES_IN_GENE
NUM_SOLUTIONS = 200


### This is the target! 
TARGET_TEXT = "An illustration of a face of a sad boy"

OUTPUT_POSTFIX = "text_sad"

#%%
### Initialize CLIP model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#%%

pgpe = PGPE(
    solution_length= SOLUTION_LENGTH,   # A solution vector has the length of 5
    popsize= NUM_SOLUTIONS,          # Our population size

    optimizer='clipup',          # Uncomment these lines if you
    optimizer_config = dict(     # would like to use the ClipUp
       max_speed=0.15,           # optimizer.
       momentum=0.9
    ),

    # optimizer='adam',            # Uncomment these lines if you
    # optimizer_config = dict(     # would like to use the Adam
    #    beta1=0.9,               # optimizer.
    #    beta2=0.999,
    #    epsilon=1e-8
    # ),
)

#%%

def draw_gene(genes):
    canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))

    num_genes = len(genes) // GENE_LENGTH
    for i in range(num_genes):
        gene = genes[i*GENE_LENGTH:(i+1)*GENE_LENGTH]
        img_index = i % NUM_IMAGES #int(abs(gene[0]) * NUM_IMAGES) % NUM_IMAGES
        x = int((gene[0] + 0.5)%1.0  * CANVAS_SIZE)
        y = int((gene[1] + 0.5)%1.0  * CANVAS_SIZE) 

        w_coef = max(0.25, pow((gene[2] + 1.0), 2.0))
        rot = gene[3] * 360.0

        im= ALPHABET_IMAGES[img_index].copy()
        im2 = im.rotate(rot, expand=True)

        # impath = imagepaths[img_index]
        org_w, org_h = im2.size
        h = int(org_h * (DEFAULT_IMG_WIDTH / org_w))
        im2.thumbnail((DEFAULT_IMG_WIDTH*w_coef, h*w_coef))
        
#        canvas.paste(im2, (x - im2.width // 2, y - im2.height // 2), im2)
        canvas.alpha_composite(im2, (x, y))    
    return canvas

def evaluate_solution(genes, returns_image = False):
    canvas = draw_gene(genes)
    
    image = preprocess(canvas).unsqueeze(0).to(device)
    text = clip.tokenize([TARGET_TEXT]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
        text_features = model.encode_text(text).cpu().numpy()

        sims = cosine_similarity(image_features, text_features)
        fitness_value = float(sims)
    
    if returns_image:
        return fitness_value, canvas
    return fitness_value


font = ImageFont.truetype("NotoMono-Regular.ttf", size=22) 
font_s = ImageFont.truetype("NotoMono-Regular.ttf", size=16) 
def draw_fitness(canvas, fitness):
    textcolor = (0, 0, 0) 
    draw = ImageDraw.Draw(canvas) 
    draw.text((20, 10), TARGET_TEXT, font=font_s, fill=textcolor)
    draw.text((20, 40), "%.3f" % fitness, font=font_s, fill=textcolor) # テキストをtextcolor(=白色)で描画
    return canvas

#%%


# Let us run the evolutionary computation for 1000 generations
for generation in range(1000):

    # Ask for solutions, which are to be given as a list of numpy arrays.
    # In the case of this example, solutions is a list which contains
    # 20 numpy arrays, the length of each numpy array being 5.
    solutions = pgpe.ask()

    # This is the phase where we evaluate the solutions
    # and prepare a list of fitnesses.
    # Make sure that fitnesses[i] stores the fitness of solutions[i].
    fitnesses = Parallel(n_jobs=-1,prefer="threads")(delayed(evaluate_solution)(genes) for genes in solutions)
    print("generate: %d average: %.4f" % (generation, np.mean(fitnesses)))
#    fitnesses = [...]  # compute the fitnesses here

    # Now we tell the result of our evaluations, fitnesses,
    # to our solver, so that it updates the center solution
    # and the spread of the search distribution.
    pgpe.tell(fitnesses)

    # Saving the best
    fitness, canvas = evaluate_solution(pgpe.center, returns_image=True)
    canvas = draw_fitness(canvas, fitness)
    path = os.path.join("./output_text", "%04d_%0.4f_%s.png" % (generation, fitness, OUTPUT_POSTFIX))
    canvas.save(path)
    if generation % 20 == 0:
        ipd.display(canvas)
    
# After 1000 generations, we print the center solution.
print(pgpe.center)


# %%

import cv2
import os
from glob import glob

video_name = './text.avi'
images = glob("./output_text/*_sadS.png")
images = sorted(images)
frame_rate = 30

frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, frame_rate, (width,height))

for image_path in images:
    video.write(cv2.imread(image_path))

#cv2.destroyAllWindows()
video.release()
# %%
!ffmpeg -i {video_name} {video_name}.mp4
# %%

# %%
