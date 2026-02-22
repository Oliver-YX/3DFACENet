python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/data/rating_face/pre_image_set" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/ijcai/beautify_demo" --k 70 --name AF357

python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/data/rating_face/pre_image_set" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/ijcai/beautify_demo" --k 100 --name AF357

python transfor_2_mesh.py --inputpath /workspace/AdBRC/xieyuan/data/demo/myfriend --savefolder /workspace/AdBRC/xieyuan/data/demo/myfriend  --k 50 --name zpw

python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/data/rating_face/pre_image_set" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/ijcai/beautify_demo" --k 0 --name AF5
python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/visualization/400meta/image/AF" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/ijcai/beautify_demo/400meta/AF" --k 90 



python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/ZAQIZABA/search_image_code/FFHQ" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ" --k 50

python beautification.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ" --k 50 --detector mtcnn --use_mica False




############# 对比方法预处理
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ/crop" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ/mask" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ/crop" --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ/mask" --save_folder  "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ/merge" 

python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA/crop" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA/mask" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA/crop" --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA/mask" --save_folder  "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA/merge" 


############ compared methods preprocessing  (FFHQ)
########## k = 90
python beautification.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/FFHQ" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ" --k 90 --detector fan --rasterizer_type pytorch3d --texture True
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/rendered_images" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/mask"
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/rendered_images"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/0/vis/rendered_images"  
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/inputs"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/input_merge" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/0/vis/rendered_images" 

cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/FFHQ/90/vis/result/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/FFHQ/OUR
cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/FFHQ/90/vis/input_merge/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/FFHQ/input


python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/Diffusion_FAE" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/Diffusion_FAE/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/StyleGAN-ADA" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/StyleGAN-ADA/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/TediGAN" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/TediGAN/mask"

python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/Diffusion_FAE"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/Diffusion_FAE/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/Diffusion_FAE/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/0/vis/rendered_images" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/StyleGAN-ADA"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/StyleGAN-ADA/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/StyleGAN-ADA/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/0/vis/rendered_images" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/TediGAN"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/TediGAN/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/FFHQ/TediGAN/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/0/vis/rendered_images"




############ compared methods preprocessing  (CelebA)
########## k = 90
python beautification.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/new" --k 90 --detector fan --rasterizer_type pytorch3d --texture True

python beautification.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/ZAQIZABA/search_image_code/CelebA" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA" --k 90 --detector fan --rasterizer_type pytorch3d --texture True
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/rendered_images" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/mask"
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/rendered_images"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images"
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/inputs"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/90/vis/input_merge" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images"

cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/CelebA/90/vis/result/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/CelebA/OUR
cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/CelebA/90/vis/input_merge/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/CelebA/input


python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Diffusion_FAE" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Diffusion_FAE/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Collaborative_Diffusion" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Collaborative_Diffusion/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/InstantID" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/InstantID/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Composable_Diffusion" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Composable_Diffusion/mask"
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Photomaker" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Photomaker/mask"

python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Diffusion_FAE"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Diffusion_FAE/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Diffusion_FAE/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Collaborative_Diffusion"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Collaborative_Diffusion/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Collaborative_Diffusion/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/InstantID"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/InstantID/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/InstantID/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images"
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Composable_Diffusion"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Composable_Diffusion/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Composable_Diffusion/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Photomaker"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Photomaker/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/vision_figure/R4_Compare_Methods/CelebA/Photomaker/result" --render_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/CelebA/0/vis/rendered_images"





python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis_new/rendered_images" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis_new/mask"
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis_new/rendered_images"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis_new/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis_new/result" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/inputs"  --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/mask" --save_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_transform/coef2mesh_demo/FFHQ/90/vis/input_merge" 

cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/CelebA/90/vis/result/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/CelebA/OUR
cp /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/flame_transform/coef2mesh_demo/CelebA/90/vis/input_merge/*.jpg  /workspace/AdBRC/xieyuan/Beautytask/3DFACENet\(IJCAI\)/vision_figure/R4_Compare_Methods/CelebA/input
