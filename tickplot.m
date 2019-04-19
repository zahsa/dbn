figure1 = figure;
XTickLabel={};

count=0;
x=0.01*2:0.01*2:.09*2;
for i=1:length(x)
    XTickLabel{i}=num2str(x(i)/2);
    count=count+1;
end;
x=[x , .2:.1:1];
for i=count+1:length(x)
    XTickLabel{i}=num2str(x(i)-.1);
    count=count+1;
end;

set(gca,'XTickLabel',XTickLabel,'XTick',x)

