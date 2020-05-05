import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as dSet
import numpy as np
import os
from tkinter import *
from tkinter.messagebox import *
import cv2
from PIL import Image, ImageTk
import PIL
from math import *
from copy import *

ABSOLUTE = 'D:/Documents/Projets

pathNormal = ABSOLUTE + "Images/Normal/"
pathAltered = ABSOLUTE + "Images/Altered/"
pathPatch = ABSOLUTE + "Images/Patch/"
pathImage = ABSOLUTE + 'Images/x128/'
pathEncode = ABSOLUTE + 'Images/Creasteph2/'
pathModels = ABSOLUTE = 'Models/'
NUMBER = 64

imageSize = 128

batchSize = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

setImages = dSet.ImageFolder(root = pathEncode, transform = transforms.Compose([transforms.RandomCrop((128, 128)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()]))
imagesLoader = torch.utils.data.DataLoader(setImages, batch_size = batchSize, shuffle = True, num_workers=0, pin_memory = True)

def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def to_img(x):
    x = x.squeeze(0)
    x = x.clamp(0, 1)
    x = x.numpy()
    x = x.transpose((1, 2, 0))
    x = x * 255
    x = x.astype(np.uint8)
    return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm2d(64),
            nn.Softplus(),

            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding=0),
            nn.BatchNorm2d(64),
            nn.Softplus(),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.Softplus(),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding=0),
            nn.BatchNorm2d(256),
            nn.Softplus(),

            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),

            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),

            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),

            #nn.BatchNorm2d(512),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size = 4, stride = 2, padding = 2),
            #nn.BatchNorm2d(512),
            nn.Softplus(),

            nn.ConvTranspose2d(512, 512, kernel_size = 5, stride = 2, padding = 0),
            nn.BatchNorm2d(512),
            nn.Softplus(),

            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 1, padding = 2),
            nn.BatchNorm2d(256),
            nn.Softplus(),

            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 0),
            nn.BatchNorm2d(128),
            nn.Softplus(),

            nn.ConvTranspose2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.Softplus(),

            nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 0),
            nn.BatchNorm2d(128),
            nn.Softplus(),

            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.Softplus(),

            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.Softplus(),

            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 2, padding = 0),
            nn.BatchNorm2d(16),
            nn.Softplus(),

            nn.ConvTranspose2d(16, 3, kernel_size = 2, stride = 1, padding = 0),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def test(self, x):
        x = self.decoder(x)
        return x
    def encode(self, x):
        x = self.encoder(x)
        return x



model = autoencoder().to(device)
model.apply(weightsInit)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr = 0.01)

im = Image.open(pathEncode + 'x128/' + 'a (0).jpg')
im = im.crop((0, 0, 128, 128))
im = np.array(im)
im = im.transpose((2, 0, 1))
im = torch.tensor(im).float().to(device)
im = im.unsqueeze(0)
x = model(im)
print(im.size())
print(model.encode(im).size())
print(x.size())

def train(number):
    global myFrame
    for i in range(number):
        if INTERFACE :
            randomScale()
            afficherPreview()
            myFrame.update()
        for data in imagesLoader:
            img, _ = data
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch : ' + str(i) + ' loss : ' + str(loss.item()))


##INTERFACE
WIDTH = 800
HEIGHT = 800
PREVIEWSIZE = 400
SLIDERLENGTH = 40
INTERFACE = True

##Sliders Hyperparamètres
COLUMNS = 40
CARACS = 8192 #Nombre de paramètres ajustables
ROWS = ceil(CARACS / COLUMNS)
#VARIABLES = np.random.rand(CARACS)


if INTERFACE :
    myFrame = Tk()
    myFrame.title('Auto-encoder')
    VARIABLES = []
    for _ in range(CARACS):
        a = DoubleVar()
        VARIABLES.append(a)
    nombre = StringVar()
    saveName = StringVar()
    meanDev = StringVar()
    meanDev.set('1')
    mean = StringVar()
    mean.set('0.5')
    numero = StringVar()
    numero.set(str(np.random.randint(0, NUMBER)))

def loadNN():
    temp = np.random.randint(0, 255, (WIDTH, HEIGHT), dtype = 'i3').astype(np.uint8)
    return ImageTk.PhotoImage(image = Image.fromarray(temp))

def rechargerImage():
    global imageCanvas
    imageCanvas.delete(ALL)
    imageCanvas.create_image(0, 0, anchor = NW, image = loadNN())

def generer():
    global VARIABLES
    global imageCanvas
    arguments = np.zeros(CARACS)
    for i in range(CARACS):
        arguments[i] = VARIABLES[i].get()
    input = arrayToTensor(arguments)
    nn.ReLU(input)
    output = model.test(input)
    output = to_img(output.cpu().data)
    global image
    imageCanvas.delete('all')
    output = np.roll(output, np.random.randint(0, 3), 2)
    image = Image.fromarray(output).resize((WIDTH, HEIGHT), PIL.Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    imageCanvas.create_image(0, 0, anchor = NW, image = img)
    imageCanvas.image = img
    """.resize((WIDTH, HEIGHT), PIL.Image.ANTIALIAS))"""

def afficherPreview():
    global imageSize
    global previewCanvas1
    global previewCanvas2
    nomb = numero.get()
    imageI = Image.open(pathEncode + 'x128/a (' + str(nomb) + ').jpg')
    cropped = imageI.crop((0, 0, 128, 128))
    initI = np.array(cropped)
    init = initI.transpose((2, 0, 1))
    init = torch.tensor(init).float()
    transforms.Normalize(init, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    init = init.to(device)
    init = init.unsqueeze(0)
    fin = model(init)
    previewCanvas1.delete('all')
    previewCanvas2.delete('all')
    image1 = Image.fromarray(initI).resize((PREVIEWSIZE, PREVIEWSIZE), PIL.Image.ANTIALIAS)
    image2 = Image.fromarray(to_img(fin.detach().cpu())).resize((PREVIEWSIZE, PREVIEWSIZE), PIL.Image.ANTIALIAS)
    photo1 = ImageTk.PhotoImage(image1)
    photo2 = ImageTk.PhotoImage(image2)
    previewCanvas1.create_image(0, 0, anchor = NW, image = photo1)
    previewCanvas2.create_image(0, 0, anchor = NW, image = photo2)
    previewCanvas1.image = photo1
    previewCanvas2.image = photo2

def randomPreview():
    numero.set(str(np.random.randint(0, NUMBER)))
    afficherPreview()

def train100():
    train(100)

def train1000():
    train(1000)

def tensorToArray(x):
    x = x.cpu().detach().numpy()

def arrayToTensor(x):
    global CARACS
    x = torch.tensor(x).float()
    x = x.view(512, 4, 4)
    #x = x.repeat(16, 1, 1)
    x.unsqueeze_(0)
    x = x.to(device)
    return x

def randomScale():
    global VARIABLES
    global meanDev
    global mean
    mD = float(meanDev.get())
    m = float(mean.get())
    for s in VARIABLES:
        s.set(np.random.normal(m, mD))
    generer()

def trainSome():
    global nombre
    train(int(nombre.get()))

def saveModel():
    global saveName
    torch.save(model.state_dict(), pathModels + saveName.get() + '.pt')

def loadModel():
    global saveName
    model.load_state_dict(torch.load(pathModels + saveName.get() + '.pt'))

def allOn():
    global VARIABLES
    for s in VARIABLES:
        s.set(1)
    generer()

def allOff():
    global VARIABLES
    for s in VARIABLES:
        s.set(0)
    generer()


if INTERFACE :
    global imageCanvas
    imageFrame = Frame(myFrame, width = WIDTH + PREVIEWSIZE, height = max(HEIGHT, 2 * PREVIEWSIZE))
    imageCanvas = Canvas(imageFrame, width = WIDTH, height = HEIGHT)
    imageCanvas.pack(side = LEFT)
    slidersFrameMaster = Frame(myFrame)
    #VARIABLES = [1, 2, 3, 4]
    ##Construction des sliders
    SLIDERS = []
    """for i in range(ROWS):
        sliderFrame = Frame(slidersFrameMaster)
        for j in range(COLUMNS):
            if i * COLUMNS + j < CARACS :
                SLIDERS.append(Scale(sliderFrame, from_ = - 1, to = 1, orient = VERTICAL, length = SLIDERLENGTH, resolution = 0.1, tickinterval = 0, width = 5, variable = VARIABLES[i * COLUMNS + j], label = '', digits  = 0, cursor = None).pack(side = LEFT))
        sliderFrame.pack()"""
    global previewFrame
    global previewCanvas1
    global previewCanvas2
    previewFrame = Frame(imageFrame, width = PREVIEWSIZE, height = 2 * PREVIEWSIZE)
    previewCanvas1 = Canvas(previewFrame, width = PREVIEWSIZE, height = PREVIEWSIZE)
    previewCanvas1.pack(side = TOP)
    previewCanvas2 = Canvas(previewFrame, width = PREVIEWSIZE, height = PREVIEWSIZE)
    previewCanvas2.pack(side = BOTTOM)
    previewFrame.pack(side = RIGHT)
    imageFrame.pack(side = TOP)

    #Boutons
    meanButton = Entry(slidersFrameMaster, textvariable = mean)
    meanButton.focus_set()
    meanButton.pack(side = LEFT)
    meanDev = Entry(slidersFrameMaster, textvariable = meanDev)
    meanDev.focus_set()
    meanDev.pack(side = LEFT)

    randomButton = Button(slidersFrameMaster, text = 'Random', command = randomScale).pack(side = LEFT)

    trainEntry = Entry(slidersFrameMaster, textvariable = nombre)
    trainEntry.focus_set()
    trainEntry.pack(side = LEFT)
    trainButton = Button(slidersFrameMaster, text = 'Train', command = trainSome).pack(side = LEFT)



    saveEntry = Entry(slidersFrameMaster, textvariable = saveName)
    saveEntry.focus_set()
    saveEntry.pack(side = LEFT)
    saveButton = Button(slidersFrameMaster, text = 'Save model', command = saveModel).pack(side = LEFT)
    loadButton = Button(slidersFrameMaster, text = 'Load model', command = loadModel).pack(side = LEFT)
    onButton = Button(slidersFrameMaster, text = 'All 1', command = allOn).pack(side = LEFT)
    offButton = Button(slidersFrameMaster, text = 'All 0', command = allOff).pack(side = LEFT)
    previewButton = Button(slidersFrameMaster, text = 'Random preview', command = randomPreview).pack(side = LEFT)
    slidersFrameMaster.pack(side = BOTTOM, padx = 10, pady = 10)

    myFrame.mainloop()
