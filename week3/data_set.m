images = (loadMNISTImages('train-images.idx3-ubyte'))';
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images1 =  (loadMNISTImages('t10k-images.idx3-ubyte'))';
labels1 =  loadMNISTLabels('t10k-labels.idx1-ubyte');
% column different images
%I = reshape(images(:,6),[28,28]);
%imshow(I)

error1 = []
rsLoss_final = [];
figure;
 %for i = 100:100:1000
 %t = templateTree('Surrogate', 'on');
 
  
 %clf = fitensemble(images(1:1000,:),labels(1:1000),'AdaBoostM2',100,'Tree')
 clf = fitensemble(images(1:1000,:), labels(1:1000), 'Bag', 1000, 'Tree','Type', 'Classification');
%clf = fitensemble(images(1:1000,:),labels(1:1000),'LPBoost',i,'Tree')
 %clf = fitensemble(images(1:1000,:),labels(1:1000),'AdaBoostM2',i,'Tree')
 rsLoss = resubLoss(clf,'Mode','Cumulative');
 %rsLoss_final = [rsLoss_final; rsLoss]
% subplot(5,2,i/100)
 plot(rsLoss);
 xlabel('Number of Learning Cycles');
 ylabel('Resubstitution Loss');
 
 [labelspredict,score] = predict(clf,images1(1:1000,:));
 error = sum((labelspredict == labels1(1:1000)))/1000
 error1 = [error1 error]
 %end
