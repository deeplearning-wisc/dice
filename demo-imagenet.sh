
echo ID: ImageNet
echo Energy
python ood_eval.py --name imagenet --in-dataset imagenet --model-arch resnet50


echo ID: ImageNet
echo DICE
python ood_eval.py --name imagenet --in-dataset imagenet --model-arch resnet50 --p 70

#echo DICE + ReACT
#python ood_eval.py --name imagenet --in-dataset imagenet --model-arch resnet50 --p 10 --clip_threshold 1.0
#python ood_eval.py --name imagenet --in-dataset imagenet --model-arch resnet50 --p 15 --clip_threshold 1.5


