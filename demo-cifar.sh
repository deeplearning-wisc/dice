
echo ID: CIFAR-100
echo No Sparsity
python ood_eval.py --in-dataset CIFAR-100
echo With Sparsity p=90
python ood_eval.py --in-dataset CIFAR-100 --p 90


echo ID: CIFAR-10
echo No Sparsity
python ood_eval.py --in-dataset CIFAR-10
echo With Sparsity p=90
python ood_eval.py --in-dataset CIFAR-10 --p 90

