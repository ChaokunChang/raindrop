from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale

def demo_transform1(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size // upscale_factor),
        ToTensor(),
    ])


def demo_transform2(crop_size=224):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def tensorlize():
    return Compose([
        ToTensor(),
    ])

def image_align(img,base=4):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img
