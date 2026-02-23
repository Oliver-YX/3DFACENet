python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/data/rating_face/pre_image_set" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/flame_transform/coef2mesh_demo/ijcai/beautify_demo" --k 70 --name AF357
python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/data/rating_face/pre_image_set" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/flame_transform/coef2mesh_demo/ijcai/beautify_demo" --k 100 --name AF357
python transfor_2_mesh.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/visualization/400meta/image/AF" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/flame_transform/coef2mesh_demo/ijcai/beautify_demo/400meta/AF" --k 90 

python beautification.py --inputpath "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ" --savefolder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/flame_transform/coef2mesh_demo/FFHQ" --k 50 --detector mtcnn --use_mica False


############# mask generation and merge image #########
python mask_generation.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ/crop" --output_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ/mask" 
python merge_image.py --input_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ/crop" --mask_folder "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ/mask" --save_folder  "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet/FFHQ/merge" 

