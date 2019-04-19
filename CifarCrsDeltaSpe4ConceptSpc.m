%recognition two supcategories of flowers and furniture

% %---neigh parametes---
% neigh=2;
% loadName=sprintf('cifarGrTrn_n%s',int2str(neigh));
% load(loadName,'cifarGrTrn');
% % cifarGrTrn=cifarGrTrn_n2;
%
% loadName=sprintf('cifarGrTst_n%s',int2str(neigh));
% load(loadName,'cifarGrTst');
% % cifarGrTst=cifarGrTst_n2;
% %-----------------------


load('cifarGrTrnEx','cifarGrTrnEx');
load('RGBPerTrn','RGBPerTrn');
load('isAnimTrn','isAnimTrn');
colmotionTr=[RGBPerTrn,isAnimTrn'];

load('meta.mat');



%---extract rows corresponding to specific classes

load('train.mat')
%flower category
crsL3=find(coarse_labels==2);
%household furniture
crsL7=find(coarse_labels==6);
crsL=[crsL3;crsL7];
DTrn=cifarGrTrnEx(crsL,:);
RGBMtrn=colmotionTr(crsL,:);

fnCat=fine_labels(crsL,:);
% fnNm=fine_label_names(fnCat+1);
%-----binary coarse labels haven't made yet
fnList3={'orchid', 'poppy', 'rose', 'sunflower', 'tulip'};
fnList7={'bed','chair','couch','table','wardrobe'};

fnList=[fnList3,fnList7];
for i=1:5
    fnI3(i)=find(strcmp(fine_label_names,fnList3{i})~=0);
    fnI7(i)=find(strcmp(fine_label_names,fnList7{i})~=0);
end

for i=1:5
    fi=find(fnCat==fnI3(i)-1);
    fnCatB(fi)=i*ones(size(fi));
end

for i=1:5
    fi=find(fnCat==fnI7(i)-1);
    fnCatB(fi)=(i+5)*ones(size(fi));
end


% fnL3=[];
% for i=1:5
%     fnL3=[fnL3;find(fine_labels==(fnI3(i)-1))];
% end

%binary target
LCTrnSpe=zeros(length(crsL3)+length(crsL7),2);
LCTrnSpe(1:length(crsL3),1)=1;
LCTrnSpe(length(crsL3)+1:end,2)=1;


LFTrnSpe=zeros(length(LCTrnSpe),10);
for i=1:length(LCTrnSpe)
    lbl=fnCatB(i);
    LFTrnSpe(i,lbl)=1;
end

% save('LFTrnSpe','LFTrnSpe');
% save('LCTrnSpe','LCTrnSpe');

clear fnCatB crsL3 crsL7 fnI3 fnI7

%-------Test-------
load('cifarGrTstEx','cifarGrTstEx');
load('RGBPerTst','RGBPerTst');
load('isAnimTst','isAnimTst');
colmotionTe=[RGBPerTst,isAnimTst'];
%---extract rows corresponding to specific classes

load('test.mat')
%flower category
crsL3=find(coarse_labels==2);
%household furniture
crsL7=find(coarse_labels==6);
crsL=[crsL3;crsL7];
DTst=cifarGrTstEx(crsL,:);
RGBMtst=colmotionTe(crsL,:);

fnCat=fine_labels(crsL,:);

% %----binary coarse labels haven't made yet
fnList3={'orchid', 'poppy', 'rose', 'sunflower', 'tulip'};
fnList7={'bed','chair','couch','table','wardrobe'};
for i=1:5
    fnI3(i)=find(strcmp(fine_label_names,fnList3{i})~=0);
    fnI7(i)=find(strcmp(fine_label_names,fnList7{i})~=0);
end

for i=1:5
    fi=find(fnCat==fnI3(i)-1);
    fnCatB(fi)=i*ones(size(fi));
end

for i=1:5
    fi=find(fnCat==fnI7(i)-1);
    fnCatB(fi)=(i+5)*ones(size(fi));
end


%binary target
LCTstSpe=zeros(length(crsL3)+length(crsL7),2);
LCTstSpe(1:length(crsL3),1)=1;
LCTstSpe(length(crsL3)+1:end,2)=1;


LFTstSpe=zeros(length(LCTstSpe),10);
for i=1:length(LCTstSpe)
    lbl=fnCatB(i);
    LFTstSpe(i,lbl)=1;
end

% save('LFTstSpe','LFTstSpe');
% save('LCTstSpe','LCTstSpe');

clear fnCatB crsL3 crsL7 fnI3 fnI7

% % netName='500-500-2000';
% % DN.layersize   = [500,500,2000];
% Cl=10;
% max_iter=5;
for iter=1:max_iter
    iter
    
    %    loadName='CifarNWRBM_DN_hBL3Ex500-500-2000_maxEp100';
    % %--RBM trained on only Art
    loadName='CifarNWGrRBM_DN_hBArt600-1000-2000_maxEp400.mat'; 
    load(loadName);
    
    DN=CifarRBM;
    
    load(loadName);
    DN=CifarRBM{iter};
    %delta
    
    w1=[DN.L{1,1}.vishid;DN.L{1,1}.hidbiases];
    w2=[DN.L{1,2}.vishid;DN.L{1,2}.hidbiases];
    w3=[DN.L{1,3}.vishid;DN.L{1,3}.hidbiases];
    
    %--coarse lables
    tr_labelsCrs=double(LCTrnSpe);
    te_labelsCrs=double(LCTstSpe);
    
    %--fine lables
    tr_labelsFn=double(LFTrnSpe);
    te_labelsFn=double(LFTstSpe);
    
    
    data = double(DTrn);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    tr_patterns1A=double(w1probs);
    
   
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    tr_patterns2A=double(w2probs);
    
%     w2probs = [w2probs  RGBMtrn]; %concatenate RGBM
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    tr_patterns3A=double(w3probs);
    
    
    data = double(DTst);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    te_patterns1A=double(w1probs);
    
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    te_patterns2A=double(w2probs);
    
%     w2probs = [w2probs  RGBMtst];    %concatenate RGBM
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    te_patterns3A=double(w3probs);
    
    
    % %--RBM trained on only Nat
    loadName='CifarNWGrRBM_DN_hBNat600-1000-2000_maxEp400.mat';
    
    load(loadName);
    DN=CifarRBM;
    
    load(loadName);
    DN=CifarRBM{iter};
    %delta
    
    w1=[DN.L{1,1}.vishid;DN.L{1,1}.hidbiases];
    w2=[DN.L{1,2}.vishid;DN.L{1,2}.hidbiases];
    w3=[DN.L{1,3}.vishid;DN.L{1,3}.hidbiases];
    
    
    data = double(DTrn);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    tr_patterns1N=double(w1probs);
    
    %concatenate RGBM
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    tr_patterns2N=double(w2probs);
    w2probs = [w2probs  RGBMtrn];
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    tr_patterns3N=double(w3probs);
    
    
    data = double(DTst);
    N=size(data,1);
    data = [data ones(N,1)];
    
    w1probs = 1./(1 + exp(-data*w1));
    te_patterns1N=double(w1probs);
    
    %concatenate RGBM
    w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    te_patterns2N=double(w2probs);
    w2probs = [w2probs  RGBMtst];
    w2probs = [w2probs  ones(N,1)];
    
    w3probs = 1./(1 + exp(-w2probs*w3));%w3probs = [w3probs  ones(N,1)];
    te_patterns3N=double(w3probs);
    
    %concatenate representations
    tr_patterns1=[tr_patterns1A,tr_patterns1N];
    tr_patterns2=[tr_patterns2A,tr_patterns2N];
    tr_patterns3=[tr_patterns3A,tr_patterns3N];
    
    te_patterns1=[te_patterns1A,te_patterns1N];
    te_patterns2=[te_patterns2A,te_patterns2N];
    te_patterns3=[te_patterns3A,te_patterns3N];
    %%%%%%
    max_epochs = 3000;
    [tr_accCrs(iter,1), te_accCrs(iter,1),CbTrnCrs1,CbTstCrs1] = DeltaClfrConf(tr_patterns1, tr_labelsCrs, te_patterns1,te_labelsCrs,max_epochs);
    [tr_accFn(iter,1), te_accFn(iter,1),CbTrnFn1,CbTstFn1] = DeltaClfrConf(tr_patterns1, tr_labelsFn, te_patterns1,te_labelsFn,max_epochs);
    
    [tr_accCrs(iter,2), te_accCrs(iter,2),CbTrnCrs2,CbTstCrs2] = DeltaClfrConf(tr_patterns2, tr_labelsCrs, te_patterns2, te_labelsCrs,max_epochs);
    [tr_accFn(iter,2), te_accFn(iter,2),CbTrnFn2,CbTstFn2] = DeltaClfrConf(tr_patterns2, tr_labelsFn, te_patterns2, te_labelsFn,max_epochs);
    
    [tr_accCrs(iter,3), te_accCrs(iter,3),CbTrnCrs3,CbTstCrs3] = DeltaClfrConf2(tr_patterns3, tr_labelsCrs, te_patterns3, te_labelsCrs,max_epochs);
    [tr_accFn(iter,3), te_accFn(iter,3),CbTrnFn3,CbTstFn3] = DeltaClfrConf2(tr_patterns3, tr_labelsFn, te_patterns3, te_labelsFn,max_epochs);
    
end



netName='Ex500-500-2000';
saveName=sprintf('NW_spe_SepSpcmaxEp200_DeltaConf%s_MaxEp%d',netName,max_epochs);

save(saveName,'tr_accCrs','te_accCrs','tr_accFn','te_accFn',...
    'CbTrnCrs1','CbTstCrs1','CbTrnFn1','CbTstFn1',...
    'CbTrnCrs2','CbTstCrs2','CbTrnFn2','CbTstFn2',...
    'CbTrnCrs3','CbTstCrs3','CbTrnFn3','CbTstFn3');