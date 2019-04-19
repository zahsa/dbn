conf_mat=CbTstFn2;
marg=18;
sum_sup_class_1=sum(sum(conf_mat(1:marg,1:marg)));
sum_sup_class_2=sum(sum(conf_mat(marg+1:end,marg+1:end)));
m=((sum_sup_class_1+sum_sup_class_2)-sum(diag(conf_mat)))/(sum(conf_mat(:)));