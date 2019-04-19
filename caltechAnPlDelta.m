load 'caltechAnPlA32Tst'
load 'caltechAnPlA32Trn'
load 'caltechAnPlA32TrnLbl'
load 'caltechAnPlA32TstLbl'

for i=1:length(DTrn32)
    DTrnCal32(i,:)=reshape(DTrn32{i},[1,32*32]);
end

for i=1:length(DTst32)
    DTstCal32(i,:)=reshape(DTst32{i},[1,32*32]);
end

allCat=max(trainCatsF);

LFTrn=zeros(length(trainCatsF),allCat);
for i=1:length(trainCatsF)
    lbl=trainCatsF(i);
    LFTrn(i,lbl)=1;
end

LFTst=zeros(length(testCatsF),allCat);
for i=1:length(testCatsF)
    lbl=testCatsF(i);
    LFTst(i,lbl)=1;
end

LCTrn=zeros(length(trainCatsH1),2);
for i=1:length(trainCatsH1)
    lbl=trainCatsH1(i);
    LCTrn(i,lbl)=1;
end

LCTst=zeros(length(testCatsH1),2);
for i=1:length(testCatsH1)
    lbl=testCatsH1(i);
    LCTst(i,lbl)=1;
end

for iter=1:max_iter
    iter
    
%     loadName='CifarNWRBM_DN_hBL3Ex500-500-2000_maxEp100';
     loadName='CifarNWRBM_DN_hB600-1000_maxEp400.mat';
    load(loadName);
    
    DN=CifarRBM;
    
    load(loadName);
    DN=CifarRBM{iter};
    %delta
    
    w1=[DN.L{1,1}.vishid;DN.L{1,1}.hidbiases];
    w2=[DN.L{1,2}.vishid;DN.L{1,2}.hidbiases];
%     w3=[DN.L{1,3}.vishid;DN.L{1,3}.hidbiases];
    
    %--coarse lables
    tr_labelsCrs=double(LCTrn);
    te_labelsCrs=double(LCTst);
    
    %--fine lables
    tr_labelsFn=double(LFTrn);
    te_labelsFn=double(LFTst);
    
    
    data = double(DTrnCal32);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    tr_patterns1=double(w1probs);
    
    %no more info
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    
    tr_patterns2=double(w2probs);
    w2probs = [w2probs  ones(N,1)];
    
%     w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
%     tr_patterns3=double(w3probs);
%     
    
    data = double(DTstCal32);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    te_patterns1=double(w1probs);
    
    %no more info
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    
    te_patterns2=double(w2probs);
    w2probs = [w2probs  ones(N,1)];
    
%     w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
%     te_patterns3=double(w3probs);
    
    max_epochs = 3000;
    [tr_accCrs(iter,1), te_accCrs(iter,1),CbTrnCrs1,CbTstCrs1] = DeltaClfrConf(tr_patterns1, tr_labelsCrs, te_patterns1,te_labelsCrs,max_epochs);
    [tr_accFn(iter,1), te_accFn(iter,1),CbTrnFn1,CbTstFn1] = DeltaClfrConf(tr_patterns1, tr_labelsFn, te_patterns1,te_labelsFn,max_epochs);
    
    [tr_accCrs(iter,2), te_accCrs(iter,2),CbTrnCrs2,CbTstCrs2] = DeltaClfrConf(tr_patterns2, tr_labelsCrs, te_patterns2, te_labelsCrs,max_epochs);
    [tr_accFn(iter,2), te_accFn(iter,2),CbTrnFn2,CbTstFn2] = DeltaClfrConf(tr_patterns2, tr_labelsFn, te_patterns2, te_labelsFn,max_epochs);
    
%     [tr_accCrs(iter,3), te_accCrs(iter,3),CbTrnCrs3,CbTstCrs3] = DeltaClfrConf2(tr_patterns3, tr_labelsCrs, te_patterns3, te_labelsCrs,max_epochs);
%     [tr_accFn(iter,3), te_accFn(iter,3),CbTrnFn3,CbTstFn3] = DeltaClfrConf2(tr_patterns3, tr_labelsFn, te_patterns3, te_labelsFn,max_epochs);
    
end

% netName='Ex500-500-2000';
% saveName=sprintf('NW_caltAP_maxEp200_DeltaConf%s_MaxEp%d',netName,max_epochs);

netName='600-1000_maxEp400';
saveName=sprintf('NW_caltAPA_DeltaConf%s_deltaEp%d',netName,max_epochs);

save(saveName,'tr_accCrs','te_accCrs','tr_accFn','te_accFn',...
    'CbTrnCrs1','CbTstCrs1','CbTrnFn1','CbTstFn1',...
    'CbTrnCrs2','CbTstCrs2','CbTrnFn2','CbTstFn2');%,...
%     'CbTrnCrs3','CbTstCrs3','CbTrnFn3','CbTstFn3');