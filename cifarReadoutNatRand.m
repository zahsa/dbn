% mot=1;
% 
% load('cifarGrTrnNat','cifarGrTrnNat');
% load('coarseLblCifarTrnNat','coarseLblCifarTrnNat');
% load('fineLblCifarTrnNat','fineLblCifarTrnNat');
% load('MoTrnNat','MoTrnNat');
% 
% % totnum=length(cifarGrTrnNat);
% % rand('state',0); %so we know the permutation of the training data
% % randomorder=randperm(totnum);
% % cifarGrTrnNat=cifarGrTrnNat(randomorder,:);
% % coarseLblCifarTrnNat=coarseLblCifarTrnNat(randomorder,:);
% % fineLblCifarTrnNat=fineLblCifarTrnNat(randomorder,:);
% 
% max_iter=1;
% load('cifartrain');
% load('meta');
% 
% load('cifarGrTstNat','cifarGrTstNat');
% load('coarseLblCifarTstNat','coarseLblCifarTstNat');
% load('fineLblCifarTstNat','fineLblCifarTstNat');
% load('MoTstNat','MoTstNat');
% 
% % totnum=length(cifarGrTstNat);
% % rand('state',0); %so we know the permutation of the training data
% % randomorder=randperm(totnum);
% % cifarGrTstNat=cifarGrTstNat(randomorder,:);
% % coarseLblCifarTstNat=coarseLblCifarTstNat(randomorder,:);
% % fineLblCifarTstNat=fineLblCifarTstNat(randomorder,:);
% 
% 
% load('cifartest');
% load('meta');
% 
% iter=1;
% 
% %--use the trained RBM to create a new representation for test and train data
% netName='600-1000-2000';epoch=400;
% if mot==1
% loadName=sprintf('CifarNWGrMotRBM_DN_hBNat%s_maxEp%d',netName,epoch);
% end
% load(loadName,'CifarRBM');
% 
% DN=CifarRBM;
% 
% w1=[DN{1,1}.L{1,1}.vishid;DN{1,1}.L{1,1}.hidbiases];
% w2=[DN{1,1}.L{1,2}.vishid;DN{1,1}.L{1,2}.hidbiases];
% w3=[DN{1,1}.L{1,3}.vishid;DN{1,1}.L{1,3}.hidbiases];
% 
% %--train data--
% 
% data = double(cifarGrTrnNat);
% N=size(data,1);
% data = [data ones(N,1)];
% 
% w1probs = 1./(1 + exp(-data*w1));
% tr_patterns1=double(w1probs);
% 
% w1probs = [w1probs  ones(N,1)];
% w2probs = 1./(1 + exp(-w1probs*w2));
% if mot == 1
%     w2probs = [w2probs  MoTrnNat];
% end
% tr_patterns2=double(w2probs);
% 
% w2probs = [w2probs  ones(N,1)];
% 
% 
% w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
% tr_patterns3=double(w3probs);
% % 
% % w3probs = [w3probs  ones(N,1)];
% % w4probs = 1./(1 + exp(-w3probs*w4));%w3probs = [w3probs  ones(N,1)];
% % tr_patterns4=double(w4probs);
% 
% %--test data---
% data = double(cifarGrTstNat);
% N=size(data,1);
% data = [data ones(N,1)];
% 
% w1probs = 1./(1 + exp(-data*w1));
% te_patterns1=double(w1probs);
% 
% w1probs = [w1probs  ones(N,1)];
% w2probs = 1./(1 + exp(-w1probs*w2));
% if mot == 1
%     w2probs = [w2probs  MoTstNat];
% end
% te_patterns2=double(w2probs);
% 
% w2probs = [w2probs  ones(N,1)];
% w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
% te_patterns3=double(w3probs);
% % 
% % w3probs = [w3probs  ones(N,1)];
% % w4probs = 1./(1 + exp(-w3probs*w4));%w3probs = [w3probs  ones(N,1)];
% % te_patterns4=double(w4probs);
% 
% max_epochs = 5000;
% 
% % %---delta
% 
% 
% [tr_accCrsNat(iter,1), te_accCrsNat(iter,1),CbTrnCrs1Nat,CbTstCrs1Nat] = DeltaClfrConf(tr_patterns1, coarseLblCifarTrnNat, te_patterns1,coarseLblCifarTstNat,max_epochs);
% [tr_accFnNat(iter,1), te_accFnNat(iter,1),CbTrnFn1Nat,CbTstFn1Nat] = DeltaClfrConf(tr_patterns1, fineLblCifarTrnNat, te_patterns1,fineLblCifarTstNat,max_epochs);
% 
% [tr_accCrsNat(iter,2), te_accCrsNat(iter,2),CbTrnCrs2Nat,CbTstCrs2Nat] = DeltaClfrConf(tr_patterns2, coarseLblCifarTrnNat, te_patterns2, coarseLblCifarTstNat,max_epochs);
% [tr_accFnNat(iter,2), te_accFnNat(iter,2),CbTrnFn2Nat,CbTstFn2Nat] = DeltaClfrConf(tr_patterns2, fineLblCifarTrnNat, te_patterns2, fineLblCifarTstNat,max_epochs);
% 
% [tr_accCrsNat(iter,3), te_accCrsNat(iter,3),CbTrnCrs3Nat,CbTstCrs3Nat] = DeltaClfrConf2(tr_patterns3, coarseLblCifarTrnNat, te_patterns3, coarseLblCifarTstNat,max_epochs);
% [tr_accFnNat(iter,3), te_accFnNat(iter,3),CbTrnFn3Nat,CbTstFn3Nat] = DeltaClfrConf2(tr_patterns3, fineLblCifarTrnNat, te_patterns3, fineLblCifarTstNat,max_epochs);
% % 
% % [tr_accCrs(iter,4), te_accCrs(iter,4),CbTrnCrs4,CbTstCrs4,er_trC4,er_teC4] = DeltaClfrConf2(tr_patterns4, CifarTrnClasLC, te_patterns4, CifarTstClasLC,max_epochs);
% % [tr_accFn(iter,4), te_accFn(iter,4),CbTrnFn4,CbTstFn4,er_trF4,er_teF4] = DeltaClfrConf2(tr_patterns4, CifarTrnClasLF, te_patterns4, CifarTstClasLF,max_epochs);


% netName=int2str(DN.layersize);
if mot==1
saveName=sprintf('NW_GRMot_NatDeltaConf_net%s_MaxIt%d',netName,max_iter);
else
saveName=sprintf('NW_GR_NatDeltaConf_net%s_MaxIt%d',netName,max_iter);
end    
save(saveName,'tr_accCrsNat','te_accCrsNat','tr_accFnNat','te_accFnNat',...
    'CbTrnCrs1Nat','CbTstCrs1Nat','CbTrnFn1Nat','CbTstFn1Nat',...
    'CbTrnCrs2Nat','CbTstCrs2Nat','CbTrnFn2Nat','CbTstFn2Nat',...
    'CbTrnCrs3Nat','CbTstCrs3Nat','CbTrnFn3Nat','CbTstFn3Nat');
%     'CbTrnCrs4','CbTstCrs4','CbTrnFn4','CbTstFn4',...
 