import os
from options.test_options import TestOptions
# from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
from util.load_mats_full import load_lm3d_full
import torch 
# from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat

import numpy as np
import  torch
import torch.nn.functional as F
from scipy.io import loadmat
# from util.load_mats import transferBFM09
from util.load_mats_full import transferBFM09
import os


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    # im_path, lm_path = get_data_path(name)
    # now_path = "/users10/lyzhang/haha/Deep3DFaceRecon_pytorch/now_evaluation/NoW_Dataset/final_release_version/iphone_pictures"
    # save_path = "/users10/lyzhang/haha/Deep3DFaceRecon_pytorch/now_evaluation/now_baseline"
    
    if opt.bfm_model=="BFM_model_front_full.mat":
        lm3d_std = load_lm3d_full(opt.bfm_folder) 
    else:   
        lm3d_std = load_lm3d(opt.bfm_folder) 
    
    # lm3d_std = load_lm3d(opt.bfm_folder) 
    
    # nofile = ["detections","multiview_expressions","multiview_neutral","multiview_occlusions","selfie"]
    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            print("%s is not found !!!"%lm_path[i])
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        model.set_input(data)  # unpack data from data loader
        
        # #########################  output all three image 
        # model.test()           # run inference 
        
        # #########################  output reconstruction image only
        model.test_rec_only()           # run inference
        # model.test_rec_lm_only()           # run inference
        # model.test_rec_only_no_exp()           # no expression
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
            save_results=True, count=i, name=img_name, add_image=False)

        model.save_mesh_real(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients
    
    # for roots, dirs, files in os.walk(now_path):
    #     for dir in dirs:
    #         if dir in nofile:
    #             continue
    #         people_name = os.path.join(now_path,dir)
    #         save_peo = save_path + "/" + str(dir)
    #         if not os.path.isdir(save_peo):
    #             os.makedirs(save_peo)
    #         for roots1, dirs1, files1 in os.walk(people_name):
    #             for dir1 in dirs1:
    #                 if dir1 == 'detections':
    #                     continue
    #                 save_peo_file = save_peo + '/' + str(dir1)
    #                 if not os.path.isdir(save_peo_file):
    #                     os.makedirs(save_peo_file)
    #                 people_file = os.path.join(people_name,dir1)
    #                 lm_path = people_file+'/'+'detections'
    #                 people = [p for p in os.listdir(people_file) if p.endswith('.jpg')]
    #                 for i in range(len(people)):
    #                     # print(i, people[i])
    #                     img_name = people[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
    #                     img_lm = lm_path + '/' + img_name + '.txt'
    #                     if not os.path.isfile(img_lm):
    #                         continue
    #                     img_people = people_file + '/' + people[i]
    #                     im_tensor, lm_tensor = read_data(img_people, img_lm, lm3d_std)
    #                     data = {
    #                         'imgs': im_tensor,
    #                         'lms': lm_tensor
    #                     }
    #                     model.set_input(data)  # unpack data from data loader
    #                     model.test_real()           # run inference
    #                     visuals = model.get_current_visuals()  # get image results
    #                     visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
    #                         save_results=True, count=i, name=img_name, add_image=False)
                        
    #                     print(save_peo_file,",",img_name)
    #                     save_obj = save_peo_file + '/' + img_name + '.obj'
    #                     save_txt = save_peo_file + '/' + img_name + '.txt'
    #                     if os.path.exists(save_txt):
    #                         continue
    #                     model.save_mesh(save_obj,save_txt)
                        
        # model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        # model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients
    

        # model.save_mesh("",img_name+'.obj'),
        #     os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.txt')) # save reconstruction meshes
        # model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt,opt.img_folder)
    
