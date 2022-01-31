
#echo ID: ImageNet
#echo No Sparsity
#echo ID: ImageNet vs Places
#
#python ood_eval_largescale.py --in_dataset imagenet2012_subset \
#                              --out_dataset places50 \
#                              --in_datadir ./datasets \
#                              --out_datadir ./datasets \
#                              --out_data_list cache/places50_selected_list.txt \
#                              --inference_k 0

#echo ID: ImageNet
#echo No Sparsity
#echo ID: ImageNet vs iNat
#
#python ood_eval_largescale.py --in_dataset imagenet2012_subset \
#                              --out_dataset inat_plantae \
#                              --in_datadir ./datasets \
#                              --out_datadir ./datasets \
#                              --out_data_list cache/inat_plantae_selected_list_nolabel.txt \
#                              --inference_k 90


#echo ID: ImageNet
#echo No Sparsity
#echo ID: ImageNet vs SUN
#
#python ood_eval_largescale.py --in_dataset imagenet2012_subset \
#                              --out_dataset sun50 \
#                              --in_datadir ./datasets \
#                              --out_datadir ./datasets \
#                              --out_data_list cache/sun50_selected_list.txt \
#                              --inference_k 90

echo ID: ImageNet
echo No Sparsity
echo ID: ImageNet vs Textures

python ood_eval_largescale.py --in_dataset imagenet2012_subset \
                              --out_dataset textures \
                              --in_datadir ./datasets \
                              --out_datadir ./datasets \
                              --inference_k 90

#
#echo With Sparsity p=90
#python ood_eval_largescale.py --in-dataset CIFAR-100 --p 90

