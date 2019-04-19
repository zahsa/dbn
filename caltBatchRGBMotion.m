
clear all


load 'caltechAnPl32Tst'
load 'caltechAnPl32Trn'
load 'caltechAnPl32TrnLbl'
load 'caltechAnPl32TstLbl'

for i=1:length(DTrn32)
    caltGrTrn(i,:)=reshape(DTrn32{i},[1,32*32]);
end

for i=1:length(DTst32)
    caltGrTst(i,:)=reshape(DTst32{i},[1,32*32]);
end

LFTrn=zeros(length(trainCatsF),12);
for i=1:length(trainCatsF)
    lbl=trainCatsF(i);
    LFTrn(i,lbl)=1;
end


LCTrn=zeros(length(trainCatsH1),2);
for i=1:length(trainCatsH1)
    lbl=trainCatsH1(i);
    LCTrn(i,lbl)=1;
end



totnum=size(caltGrTrn,1);
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);
batchsize = 10;
numbatches=totnum/batchsize;
numdims  =  size(caltGrTrn,2);
Cl=12;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, Cl, numbatches);
batchdata=[];batchtargets1=[];batchtargets2=[];
for b=1:numbatches
    batchdata(:,:,b) = caltGrTrn(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets1(:,:,b) = LCTrn(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets2(:,:,b) = LFTrn(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end


batchTrnGrCalt=batchdata;
batchTrnLblCrCalt=batchtargets1;
batchTrnLblFnCalt=batchtargets2;

save('batchTrnGrCalt','batchTrnGrCalt');
save('batchTrnLblCrCalt','batchTrnLblCrCalt');
save('batchTrnLblFnCalt','batchTrnLblFnCalt');

% %---make the extra visual information into batches
% load('RGBPerTrn','RGBPerTrn');
% load('isAnimTrn','isAnimTrn');
% colmotionTr=[RGBPerTrn,isAnimTrn'];
% 
% % numdims  =  size(RGBPerTrn,2);
% % numdims = numdims + size(isAnimTrn,1);
% numdims = 4;
% batchColMoTr = zeros(batchsize, numdims, numbatches);
% 
% for b=1:numbatches
%     batchColMoTr(:,:,b) = colmotionTr(randomorder(1+(b-1)*batchsize:b*batchsize), :);
% end
% 
% save('batchColMoTr','batchColMoTr');


