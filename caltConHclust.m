% load('NW_GRGBMo_lr_DeltaConf_MaxIt1');
load('NW_caltAP_DeltaConf600-1000_maxEp400_deltaEp3000');

%normalize confusion mat
CbTstFn1n=CbTstFn1./repmat(sum(CbTstFn1,2),[1,size(CbTstFn1,2)]);
CbTstFn2n=CbTstFn2./repmat(sum(CbTstFn2,2),[1,size(CbTstFn2,2)]);

% CbTstFn1n(CbTstFn1n<=.05)=0;
figure;imagesc(CbTstFn1n);colormap gray
figure;imagesc(CbTstFn2n);colormap gray


offDiagAn=sum(sum(CbTstFn2n(1:18,1:18)))-sum(diag(CbTstFn2n(1:18,1:18)))/18*18;
offDiagPl=sum(sum(CbTstFn2n(19:end,19:end)))-sum(diag(CbTstFn2n(19:end,19:end)))/7*7;
offDiagOthB=sum(sum(CbTstFn2n(1:18,19:end)))-sum(diag(CbTstFn2n(1:18,19:end)))/18*7;
offDiagOthT=sum(sum(CbTstFn2n(19:end,1:18)))-sum(diag(CbTstFn2n(19:end,1:18)))/18*7;

(offDiagAn+offDiagPl)

ccr_tst=sum(diag(CbTstFn3))/sum(sum(CbTstFn3));

% xLbl=catNamesLbl;

for i=1:length(catNamesLbl)
    xLbl{i}=catNamesLbl(i).name;
end

set(gca,'XTick',1:numel(xLbl),...                         %# Change the axes tick marks
        'XTickLabel',xLbl,...  %#   and tick labels
        'YTick',1:numel(xLbl),...
        'YTickLabel',xLbl,...
        'TickLength',[0 0]);
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
text(b,repmat(26,length(c),1),a,'HorizontalAlignment','right','rotation',rot,'fontsize',7);


%---Hclust on layer one representation
eucliD = pdist(CbTstFn2,'euclidean');
linkEucli3 = linkage(eucliD,'ward');
thresh=100;figure;[~,~,permEucliWard3]=dendrogram(linkEucli3,0,'labels',xLbl,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 3');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(8,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH3=coarse_label_names(permEucliWard3);



eucliD = pdist(CbTstCrs2,'euclidean');
linkEucli2 = linkage(eucliD,'ward');
thresh=100;figure;[~,~,permEucliWard2]=dendrogram(linkEucli2,0,'labels',coarse_label_names,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 2');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(20,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH2=coarse_label_names(permEucliWard2);

eucliD = pdist(CbTstCrs1,'euclidean');
linkEucli1 = linkage(eucliD,'ward');
thresh=100;figure;[~,~,permEucliWard1]=dendrogram(linkEucli1,0,'labels',coarse_label_names,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 1');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(20,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH1=coarse_label_names(permEucliWard1);


labels=coarse_label_names;
set(h,'FontSize',6)
set(gca,'YTickLabel',labels,'YTick',1:numel(labels))
set(gca,'XTickLabel',labels,'XTick',1:numel(labels))

a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=60;
th=text(b,repmat(size(labels,1),length(c),1),a,'HorizontalAlignment','right','rotation',rot,'fontsize',7);


figure;imagesc(CbTrnFn1)
figure;imagesc(CbTrnFn2)
figure;imagesc(CbTrnFn3)

figure;imagesc(CbTstFn1)
figure;imagesc(CbTstFn2)
figure;imagesc(CbTstFn3)

labels=fine_label_names;
set(gca,'FontSize',6)
set(gca,'YTickLabel',labels,'YTick',1:numel(labels))
set(gca,'XTickLabel',labels,'XTick',1:numel(labels))

a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=60;
th=text(b,repmat(size(labels,1),length(c),1),a,'HorizontalAlignment','right','rotation',rot,'fontsize',7);


%---Hclust on layer one representation
eucliD = pdist(CbTstFn3,'euclidean');
linkEucli3 = linkage(eucliD,'ward');
thresh=100;figure;[~,~,permEucliWard3]=dendrogram(linkEucli3,0,'labels',coarse_label_names,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 3');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(20,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH3=coarse_label_names(permEucliWard3);



eucliD = pdist(CbTstFn2,'euclidean');
linkEucli2 = linkage(eucliD,'ward');
thresh=20;figure;[~,~,permEucliWard2]=dendrogram(linkEucli2,0,'labels',fine_label_names,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 2');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(7,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH2=fine_label_names(permEucliWard2);

eucliD = pdist(CbTstFn1,'euclidean');
linkEucli1 = linkage(eucliD,'ward');
thresh=100;figure;[~,~,permEucliWard1]=dendrogram(linkEucli1,0,'labels',fine_label_names,'colorthreshold',thresh);title('hclust on eucliD of confusion mat of layer 1');
set(gca,'FontSize',7)
a=get(gca,'XTickLabel');
set(gca,'XTickLabel',[]);
b=get(gca,'XTick');
c=get(gca,'YTick');
rot=90;
th=text(b,repmat(1,length(b),1),a,'HorizontalAlignment','right','rotation',rot,'FontSize',7);
ordH1=fine_label_names(permEucliWard1);

%---delta error
figure;plot(er_teF3,'r');hold on;plot(er_trF3,'b');
figure;plot(er_teC3,'r');hold on;plot(er_trC3,'b');

teC=[er_teC3(1),er_teC3(end)]

teF=[er_teF3(1),er_teF3(end)]