from collections import OrderedDict
import numpy as np
import torch
import architecture as arch
import runway
from tiles import ImageSlicer
from skimage.transform import resize

if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')


@runway.setup(options={'interpolationFactor': runway.number(default=0.8, min=0, max=1, step=0.01)})
def setup(opts):
    net_PSNR_path = './models/RRDB_PSNR_x4.pth'
    net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'
    net_PSNR = torch.load(net_PSNR_path)
    net_ESRGAN = torch.load(net_ESRGAN_path)
    net_interp = OrderedDict()
    interpolation_factor = opts.get('interpolationFactor', 0.5)
    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - interpolation_factor) * v_PSNR + interpolation_factor * v_ESRGAN
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(net_interp)
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model

def process_tile(model, tile):
    img = np.transpose(tile[:, :, [2, 1, 0]], (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = resize(output, tile.shape)
    return output

@runway.command(name='upscale', inputs={'image': runway.image}, outputs={'upscaled': runway.image})
def upscale(model, inputs):
    img = np.array(inputs['image'])
    img = img * 1.0 / 255
    tiler = ImageSlicer(img.shape, tile_size=(512, 512), tile_step=(256, 256), weight='pyramid')
    tiles = [process_tile(model, tile) for tile in tiler.split(img)]
    output = tiler.merge(tiles)
    output = (output * 255.0).round().astype('uint8')
    return dict(upscaled=output)

if __name__ == '__main__':
    runway.run(port=4323)
