%--with mot inf


load('cifarGrTrnEx','cifarGrTrnEx');
% load('RGBPerTrn','RGBPerTrn');
load('isAnimTrn','isAnimTrn');
motionTr=[isAnimTrn'];

load('meta.mat');



%---extract rows corresponding to specific classes

load('train.mat')
%flower category
% crsL3=find(coarse_labels==2);
%household furniture
crsL7=find(coarse_labels==6);

DTrn7=cifarGrTrnEx(crsL7,:);

Mtrn7=motionTr(crsL7,:);

fnCat7=fine_labels(crsL7,:);

% fnNm=fine_label_names(fnCat+1);
%-----binary coarse labels haven't made yet
% fnList3={'orchid', 'poppy', 'rose', 'sunflower', 'tulip'};
fnList7={'bed','chair','couch','table','wardrobe'};

for i=1:5
    fnI7(i)=find(strcmp(fine_label_names,fnList7{i})~=0);
end

for i=1:5
    fi=find(fnCat7==fnI7(i)-1);
    fnCatB7(fi)=i*ones(size(fi));
end




LFTrn7=zeros(length(fnI7),5);
for i=1:length(fnCat7)
    lbl=fnCatB7(i);
    LFTrn7(i,lbl)=1;
end

% save('LFTrnSpe','LFTrnSpe');
% save('LCTrnSpe','LCTrnSpe');

clear fnCatB crsL7 fnI7 

%-------Test-------
load('cifarGrTstEx','cifarGrTstEx');
% load('RGBPerTst','RGBPerTst');
load('isAnimTst','isAnimTst');
% colmotionTe=[RGBPerTst,isAnimTst'];
motionTe=[isAnimTst'];

%---extract rows corresponding to specific classes

load('test.mat')
%flower category
crsL7=find(coarse_labels==2);
%household furniture
% crsL7=find(coarse_labels==6);

DTst7=cifarGrTstEx(crsL7,:);
Mtst7=motionTe(crsL7,:);

fnCat7=fine_labels(crsL7,:);

% %----binary coarse labels haven't made yet
% fnList3={'orchid', 'poppy', 'rose', 'sunflower', 'tulip'};
fnList7={'bed','chair','couch','table','wardrobe'};
for i=1:5
    fnI7(i)=find(strcmp(fine_label_names,fnList7{i})~=0);
end

for i=1:5
    fi=find(fnCat7==fnI7(i)-1);
    fnCatB7(fi)=i*ones(size(fi));
end


LFTst7=zeros(length(fnI7),5);
for i=1:length(fnCat7)
    lbl=fnCatB7(i);
    LFTst7(i,lbl)=1;
end

% save('LFTstSpe','LFTstSpe');
% save('LCTstSpe','LCTstSpe');

clear fnCatB crsL7  fnI7 

% % netName='500-500-2000';
% % DN.layersize   = [500,500,2000];
% Cl=10;
% max_iter=5;
for iter=1:max_iter
    iter
    
%     %--RBM trained on Art+Nat
%      loadName='CifarNWRBM_DN_hB_L3Gr500-500-2000_maxEp100.mat';

% %--RBM trained on only Art
%     loadName='CifarNWGrMoRBM_DN_hBArt600-1000-2000_maxEp400';
 
%--RBM trained on Nat
    loadName='CifarNWGrMotRBM_DN_hBNat600-1000-2000_maxEp400';

    load(loadName);
    DN=CifarRBM;
    
    load(loadName);
    DN=CifarRBM{iter};
    
    
%     %--coarse lables
%     tr_labelsCrs=double(LCTrn7);
%     te_labelsCrs=double(LCTst7);
    
    %--fine lables
    tr_labelsFn=double(LFTrn7);
    te_labelsFn=double(LFTst7);
    
    
    w1=[DN.L{1,1}.vishid;DN.L{1,1}.hidbiases];
    w2=[DN.L{1,2}.vishid;DN.L{1,2}.hidbiases];
    w3=[DN.L{1,3}.vishid;DN.L{1,3}.hidbiases];
    
    data = double(DTrn7);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    tr_patterns1=double(w1probs);
    
    %concatenate RGBM
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    tr_patterns2=double(w2probs);
    w2probs = [w2probs  Mtrn7];
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    tr_patterns3=double(w3probs);
    
    
    data = double(DTst7);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    te_patterns1=double(w1probs);
    
    %concatenate RGBM
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    te_patterns2=double(w2probs);
    w2probs = [w2probs  Mtst7];
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    te_patterns3=double(w3probs);
    
    max_epochs = 3000;
%     [tr_accCrs(iter,1), te_accCrs(iter,1),CbTrnCrs1,CbTstCrs1] = DeltaClfrConf(tr_patterns1, tr_labelsCrs, te_patterns1,te_labelsCrs,max_epochs);
    [tr_accFn(iter,1), te_accFn(iter,1),CbTrnFn1,CbTstFn1] = DeltaClfrConf(tr_patterns1, tr_labelsFn, te_patterns1,te_labelsFn,max_epochs);
    
%     [tr_accCrs(iter,2), te_accCrs(iter,2),CbTrnCrs2,CbTstCrs2] = DeltaClfrConf(tr_patterns2, tr_labelsCrs, te_patterns2, te_labelsCrs,max_epochs);
    [tr_accFn(iter,2), te_accFn(iter,2),CbTrnFn2,CbTstFn2] = DeltaClfrConf(tr_patterns2, tr_labelsFn, te_patterns2, te_labelsFn,max_epochs);
    
%     [tr_accCrs(iter,3), te_accCrs(iter,3),CbTrnCrs3,CbTstCrs3] = DeltaClfrConf2(tr_patterns3, tr_labelsCrs, te_patterns3, te_labelsCrs,max_epochs);
    [tr_accFn(iter,3), te_accFn(iter,3),CbTrnFn3,CbTstFn3] = DeltaClfrConf2(tr_patterns3, tr_labelsFn, te_patterns3, te_labelsFn,max_epochs);
    
end


%--save delta results on Art representation
% netName='Ex500-500-2000';
% saveName=sprintf('NW_ArtFn_DNArt_DeltaConf%s_DeltaMaxEp%d',netName,max_epochs);

%--save delta results on Art+Nat representation
netName='600-1000-2000';
saveName=sprintf('NW_ArtFn_DNNat_DeltaConf%s_DeltaMaxEp%d',netName,max_epochs);

save(saveName,'tr_accFn','te_accFn',...
    'CbTrnFn1','CbTstFn1',...
    'CbTrnFn2','CbTstFn2',...
    'CbTrnFn3','CbTstFn3');