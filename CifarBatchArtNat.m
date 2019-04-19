% D=data(1,:);
% rd=reshape(D,[32,32,3]);
% imshow(rgb2gray(reshape(D,[32,32,3])))
clear all


%---extract rows corresponding to Art/Nat classes--
load('train.mat')
load('targetCrsTrn','targetCrsTrn')
load('targetFnTrn','targetFnTrn')
load('isAnimTrn','isAnimTrn');

isAnimTrn=isAnimTrn';

crsArt=[4,6,7,10,19,20];
DTrnArt=[];LFTrnArt=[];LCTrnArt=[];MoTrnArt=[];
for crsLbl=crsArt
    inds=find(coarse_labels==crsLbl-1);
    DTrnArt=[DTrnArt;data(inds,:)];
    MoTrnArt=[MoTrnArt;isAnimTrn(inds)];
    LFTrnArt=[LFTrnArt;targetFnTrn(inds,:);];
    LCTrnArt=[LCTrnArt;targetCrsTrn(inds,:);];
end

crsNat=[1,2,3,5,8,9,11,12,13,14,15,16,17,18];
DTrnNat=[];LFTrnNat=[];LCTrnNat=[];MoTrnNat=[];
for crsLbl=crsNat
    inds=find(coarse_labels==crsLbl-1);
    DTrnNat=[DTrnNat;data(inds,:)];
    MoTrnNat=[MoTrnNat;isAnimTrn(inds)];
    LFTrnNat=[LFTrnNat;targetFnTrn(inds,:);];
    LCTrnNat=[LCTrnNat;targetCrsTrn(inds,:);];
end


%-----
Cl= 100;
cifarGrTrnArt=[];

for i=1:size(DTrnArt,1)
    grayData=rgb2gray(reshape(DTrnArt(i,:),[32,32,3]));
    dgrayData=double(grayData);
%     im=(dgrayData-min(dgrayData(:)))/(max(dgrayData(:))-min(dgrayData(:)));
    %  imshow(grayI)
    cifarGrTrnArt(i,:)=reshape(dgrayData,[1,32*32]);
end
coarseLblCifarTrnArt=LCTrnArt;
fineLblCifarTrnArt=LFTrnArt;
cifarGrTrnArt=cifarGrTrnArt/255; %--according to mnist?

save('cifarGrTrnArt','cifarGrTrnArt');
save('coarseLblCifarTrnArt','coarseLblCifarTrnArt');
save('fineLblCifarTrnArt','fineLblCifarTrnArt');
save('MoTrnArt','MoTrnArt');

cifarGrTrnNat=[];

for i=1:size(DTrnNat,1)
    grayData=rgb2gray(reshape(DTrnNat(i,:),[32,32,3]));
    dgrayData=double(grayData);
%     im=(dgrayData-min(dgrayData(:)))/(max(dgrayData(:))-min(dgrayData(:)));
    %  imshow(grayI)
    cifarGrTrnNat(i,:)=reshape(dgrayData,[1,32*32]);
end
coarseLblCifarTrnNat=LCTrnNat;
fineLblCifarTrnNat=LFTrnNat;
cifarGrTrnNat=cifarGrTrnNat/255; %--according to mnist?

save('cifarGrTrnNat','cifarGrTrnNat');
save('coarseLblCifarTrnNat','coarseLblCifarTrnNat');
save('fineLblCifarTrnNat','fineLblCifarTrnNat');
save('MoTrnNat','MoTrnNat');

clear grayData dgrayData  

%-batch

%--ArtTrn
totnum=length(cifarGrTrnArt);
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);
batchsize = 100;
numbatches=totnum/batchsize;
numdims  =  size(cifarGrTrnArt,2);

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, Cl, numbatches);
batchdata=[];batchtargets1=[];batchtargets2=[];
for b=1:numbatches
    batchdata(:,:,b) = cifarGrTrnArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets1(:,:,b) = coarseLblCifarTrnArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets2(:,:,b) = fineLblCifarTrnArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end

batchTrnGrCifarArt=batchdata;
batchTrnLblCrCifarArt=batchtargets1;
batchTrnLblFnCifarArt=batchtargets2;

save('batchTrnGrCifarArt','batchTrnGrCifarArt');
save('batchTrnLblCrCifarArt','batchTrnLblCrCifarArt');
save('batchTrnLblFnCifarArt','batchTrnLblFnCifarArt');

numdims = 1;
batchMoTrArt = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
    batchMoTrArt(:,:,b) = MoTrnArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
save('batchMoTrArt','batchMoTrArt');

%--NatTrn
totnum=length(cifarGrTrnNat);
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);
batchsize = 100;
numbatches=totnum/batchsize;
numdims  =  size(cifarGrTrnNat,2);

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, Cl, numbatches);
batchdata=[];batchtargets1=[];batchtargets2=[];
for b=1:numbatches
    batchdata(:,:,b) = cifarGrTrnNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets1(:,:,b) = coarseLblCifarTrnNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets2(:,:,b) = fineLblCifarTrnNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end

batchTrnGrCifarNat=batchdata;
batchTrnLblCrCifarNat=batchtargets1;
batchTrnLblFnCifarNat=batchtargets2;

save('batchTrnGrCifarNat','batchTrnGrCifarNat');
save('batchTrnLblCrCifarNat','batchTrnLblCrCifarNat');
save('batchTrnLblFnCifarNat','batchTrnLblFnCifarNat');

numdims = 1;
batchMoTrNat = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
    batchMoTrNat(:,:,b) = MoTrnNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
save('batchMoTrNat','batchMoTrNat');

%-test data
load('test.mat')
load('targetCrsTst','targetCrsTst')
load('targetFnTst','targetFnTst')
load('isAnimTst','isAnimTst');

isAnimTst=isAnimTst';

DTstArt=[];LFTstArt=[];LCTstArt=[];MoTstArt=[];
for crsLbl=crsArt
    inds=find(coarse_labels==crsLbl-1);
    DTstArt=[DTstArt;data(inds,:)];
    MoTstArt=[MoTstArt;isAnimTst(inds)];
    LFTstArt=[LFTstArt;targetFnTst(inds,:);];
    LCTstArt=[LCTstArt;targetCrsTst(inds,:);];
end

DTstNat=[];LFTstNat=[];LCTstNat=[];MoTstNat=[];
for crsLbl=crsNat
    inds=find(coarse_labels==crsLbl-1);
    DTstNat=[DTstNat;data(inds,:)];
    MoTstNat=[MoTstNat;isAnimTst(inds)];
    LFTstNat=[LFTstNat;targetFnTst(inds,:);];
    LCTstNat=[LCTstNat;targetCrsTst(inds,:);];
end

%--Art
cifarGrTst=[];
for i=1:size(DTstArt,1)
    grayData=rgb2gray(reshape(DTstArt(i,:),[32,32,3]));
    dgrayData=double(grayData);
%     im=(dgrayData-min(dgrayData(:)))/(max(dgrayData(:))-min(dgrayData(:)));
    %  imshow(im)
    cifarGrTstArt(i,:)=reshape(dgrayData,[1,32*32]);
end
cifarGrTstArt=cifarGrTstArt/255; %--according to mnist?
coarseLblCifarTstArt=LCTstArt;
fineLblCifarTstArt=LFTstArt;
save('cifarGrTstArt','cifarGrTstArt');
save('coarseLblCifarTstArt','coarseLblCifarTstArt');
save('fineLblCifarTstArt','fineLblCifarTstArt');
save('MoTstArt','MoTstArt');

%--Nat
cifarGrTst=[];
for i=1:size(DTstNat,1)
    grayData=rgb2gray(reshape(DTstNat(i,:),[32,32,3]));
    dgrayData=double(grayData);
%     im=(dgrayData-min(dgrayData(:)))/(max(dgrayData(:))-min(dgrayData(:)));
    %  imshow(im)
    cifarGrTstNat(i,:)=reshape(dgrayData,[1,32*32]);
end
cifarGrTstNat=cifarGrTstNat/255; %--according to mnist?
coarseLblCifarTstNat=LCTstNat;
fineLblCifarTstNat=LFTstNat;
save('cifarGrTstNat','cifarGrTstNat');
save('coarseLblCifarTstNat','coarseLblCifarTstNat');
save('fineLblCifarTstNat','fineLblCifarTstNat');
save('MoTstNat','MoTstNat');


%test_batch

%--ArtTst
totnum=size(cifarGrTstArt,1);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(cifarGrTstArt,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, Cl, numbatches);

for b=1:numbatches
    testbatchdata(:,:,b) = cifarGrTstArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets1(:,:,b) = coarseLblCifarTstArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets2(:,:,b) = fineLblCifarTstArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
batchTstGrCifarArt=testbatchdata;
batchTstLblCrCifarArt=batchtargets1;
batchTstLblFnCifarArt=batchtargets2;

save('batchTstGrCifarArt','batchTstGrCifarArt');
save('batchTstLblCrCifarArt','batchTstLblCrCifarArt');
save('batchTstLblFnCifarArt','batchTstLblFnCifarArt');

numdims = 1;
batchMoTeArt = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
    batchMoTeArt(:,:,b) = MoTstArt(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
save('batchMoTeArt','batchMoTeArt');

% %--NatTst
totnum=size(cifarGrTstNat,1);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(cifarGrTstNat,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, Cl, numbatches);

for b=1:numbatches
    testbatchdata(:,:,b) = cifarGrTstNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets1(:,:,b) = coarseLblCifarTstNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets2(:,:,b) = fineLblCifarTstNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
batchTstGrCifarNat=testbatchdata;
batchTstLblCrCifarNat=batchtargets1;
batchTstLblFnCifarNat=batchtargets2;

save('batchTstGrCifarNat','batchTstGrCifarNat');
save('batchTstLblCrCifarNat','batchTstLblCrCifarNat');
save('batchTstLblFnCifarNat','batchTstLblFnCifarNat');

numdims = 1;
batchMoTeNat = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
    batchMoTeNat(:,:,b) = MoTstNat(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
save('batchMoTeNat','batchMoTeNat');
