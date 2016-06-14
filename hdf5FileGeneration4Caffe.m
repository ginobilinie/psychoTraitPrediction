function hdf5FileGeneration4Caffe()

% load the dataset, this is built in to matlab

mat=importdata('BFI44_A.txt');
data=mat(:,1:end-1);
labels=mat(:,end);
labels=(labels-min(labels))/(max(labels)-min(labels));
% the data is ordered, we want to randomly select 100 points for train, 50
% for test. This part just generates a random list of array indices.
len=size(data,1);
featureLen=size(data,2);
indices = randperm(len);
train_indices = indices(1:len-100);
test_indices = indices(len-100+1:end);


% meas contains the features, we use our random indices to select a subset
% for training and testing
train_data = data(train_indices,:);
test_data = data(test_indices,:);
% species is a cell array of strings, we need to convert to integers. 
% casting to int16 creates numbers 1-3, we subtract 1 so we get 0-2.
% caffe likes labels that start at 0.
train_labels = labels(train_indices);
test_labels = labels(test_indices);

% now our train and test sets have been made we need to write them to HDF5
% files. If the files exist, delete them.
delete('psA_train.hdf5')
delete('psA_test.hdf5')
% First write the train data
h5create('psA_train.hdf5','/data',[featureLen,len-100],'Datatype','double');
h5write('psA_train.hdf5','/data',train_data');
h5create('psA_train.hdf5','/label',[1,len-100],'Datatype','double');
h5write('psA_train.hdf5','/label',train_labels');
% now write the test data
h5create('psA_test.hdf5','/data',[featureLen,100],'Datatype','double');
h5write('psA_test.hdf5','/data',test_data');
h5create('psA_test.hdf5','/label',[1,100],'Datatype','double');
h5write('psA_test.hdf5','/label',test_labels');

% our datasets have now been created, now we can run caffe
return