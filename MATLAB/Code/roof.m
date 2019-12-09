%Importing the images.
addpath(genpath('stac')); % generating path and setting it as working directory
regionNames = ["borde_rural", "borde_soacha", "mixco_1_and_ebenezer", "mixco_3", "castries", "dennery", "gros_islet"]; %Regions to be processed


for idx = 1:numel(regionNames)
    path = which(regionNames(idx) + '_ortho-cog.tif');
    bimg = bigimage(which(path)); %creating BigTIFF image of each region
    
    brgb(idx) = apply(bimg, 1, @separateChannels , 'UseParallel',true); %splitting image into RGB channels and mask channel of opacity
    
    fid = fopen(regionNames(idx) + "-imagery.json"); 
    imageryStructs(idx) = jsondecode(fread(fid,inf,'*char')');
    fclose(fid);
    for k = 1:numel(brgb(idx).SpatialReferencing)
        brgb(idx).SpatialReferencing(k).XWorldLimits = [imageryStructs(idx).bbox(1) imageryStructs(idx).bbox(3)]; % Longitude limits
        brgb(idx).SpatialReferencing(k).YWorldLimits = [imageryStructs(idx).bbox(2) imageryStructs(idx).bbox(4)]; % Latitude limits
    end
end
clear bimg

%Creating Training Data
for idx = 1:numel(regionNames)    
    fid = fopen("train-" + regionNames(idx) + ".geojson");
    trainingStructs(idx) = jsondecode(fread(fid,inf,'*char')'); %parsing the GeoJSON file 
    fclose(fid);
end

numTrainRegions = arrayfun(@(x)sum(length(x.features)), trainingStructs); %Creating a lambda and applying it on training data
numTrainRegionsCumulative = cumsum(numTrainRegions); %used to denote the no.of. observations before the particular region
numTrain = sum(numTrainRegions);%Summing up no.of structures in all training regions
trainingStruct = cat(1, trainingStructs.features); %Concatinating all the data

trainID = cell(numTrain,1);         % Training data ID
trainMaterial = cell(numTrain,1);   % Training data material
trainCoords = cell(numTrain,1);     % Training data coordinates

regionIdx = 1;
for k = 1:numTrain
    trainID{k} = trainingStruct(k).id; %creating the id column
    trainMaterial{k} = trainingStruct(k).properties.roof_material; %creating material column
    coords = trainingStruct(k).geometry.coordinates;
    disp(coords)
    if iscell(coords) %extracting data if it was an array
        coords = coords{1};
    end
    trainCoords{k} = squeeze(coords);%putting the y co-ordinates adjacent to the x co-ordinates    
    if k > numTrainRegionsCumulative(regionIdx)
        regionIdx = regionIdx + 1;
    end
    trainCoords{k}(:,2) = brgb(regionIdx).SpatialReferencing(1).YWorldLimits(2)-(trainCoords{k}(:,2)-brgb(regionIdx).SpatialReferencing(1).YWorldLimits(1)); %shifting origin
end

trainMaterial = categorical(trainMaterial); %Converting text array to categorical array
clear trainingStruct trainingStructs %clearing the memory

display = false
displayRegion = "borde_rural";
displayRegionNum = find(regionNames==displayRegion);

if  display
    if displayRegionNum == 1
        polyIndices = 1:numTrainRegions(displayRegionNum);
    else
        polyIndices = numTrainRegions(displayRegionNum-1) + 1:numTrainRegions(displayRegionNum);
    end%determining the no. of indices in the polygon
    
    polyFcn = @(position) images.roi.Polygon('Position',position);% Creating a function to extract the co-ordinates
    polys = cellfun(polyFcn,trainCoords(polyIndices));%applying the above function

    bigimageshow(brgb(displayRegionNum))
    xlabel('Longitude')
    ylabel('Latitude')
    set(polys,'Visible','on')
    set(polys,'Parent',gca)
    set(polys,'Color','r')
    
    figure
    displayIndices = randi(numTrainRegions(displayRegionNum),4,1);
    for k = 1:numel(displayIndices)
        coords = trainCoords{displayIndices(k) + polyIndices(1) - 1};
        regionImg = getRegion(brgb(displayRegionNum),1,[min(coords(:,1)) min(coords(:,2))],[max(coords(:,1)) max(coords(:,2))]);
        subplot(2,2,k)
        imshow(regionImg);
    end
end

if exist("training_data","dir")  
    load(fullfile("training_data","training_data")); %loading the dataset if it already exists

else
    mkdir("training_data")
    cd training_data
    materialCategories = categories(trainMaterial);
    for k = 1:numel(materialCategories)
        mkdir(materialCategories{k})
    end
    cd ..

    regionIdx = 1;
    for k = 1:numTrain
        if k > numTrainRegionsCumulative(regionIdx)
            regionIdx = regionIdx + 1;
        end
        coords = trainCoords{k};
        regionImg = getRegion(brgb(regionIdx),1,[min(coords(:,1)) min(coords(:,2))],[max(coords(:,1)) max(coords(:,2))]); % cutting the building images out of the parent image
        imgFilename = fullfile("training_data", string(trainMaterial(k)) , trainID{k}+".png" ); %naming the image
        imwrite(regionImg,imgFilename); %classifying the images into their roof materials folders
    end 

    save(fullfile("training_data","training_data"),"trainID","trainMaterial","trainCoords")

end


if exist("test_data","dir")  

    load(fullfile("test_data","test_data"));

else
    mkdir("test_data")
    regionNames = ["borde_rural","borde_soacha","mixco_1_and_ebenezer","mixco_3","dennery"];
    for idx = 1:numel(regionNames)
        bimg = bigimage(which(regionNames(idx) + "_ortho-cog.tif"));
        brgb(idx) = apply(bimg,1, @separateChannels,'UseParallel',true);
        fid = fopen(regionNames(idx) + "-imagery.json");
        imageryStructs(idx) = jsondecode(fread(fid,inf,'*char')');
        fclose(fid);
        for k = 1:numel(brgb(idx).SpatialReferencing)
            brgb(idx).SpatialReferencing(k).XWorldLimits = [imageryStructs(idx).bbox(1) imageryStructs(idx).bbox(3)]; % Longitude limits
            brgb(idx).SpatialReferencing(k).YWorldLimits = [imageryStructs(idx).bbox(2) imageryStructs(idx).bbox(4)]; % Latitude limits
        end
    end
    clear bimg

    for idx = 1:numel(regionNames)
        fid = fopen("test-" + regionNames(idx) + ".geojson");
        testStructs(idx) = jsondecode(fread(fid,inf,'*char')');
        fclose(fid);
    end
    
    numTestRegions = arrayfun(@(x)sum(length(x.features)), testStructs);
    numTestRegionsCumulative = cumsum(numTestRegions);
    numTest = sum(numTestRegions);
    testStruct = cat(1, testStructs.features);
    testID = cell(numTest,1);         % Test data ID
    testCoords = cell(numTest,1);     % Test data coordinates
    regionIdx = 1;
    for k = 1:numTest
        testID{k} = testStruct(k).id;
        coords = testStruct(k).geometry.coordinates;
        if iscell(coords)
            coords = coords{1};
        end
        testCoords{k} = squeeze(coords);
        if k > numTestRegionsCumulative(regionIdx)
            regionIdx = regionIdx + 1;
        end
        testCoords{k}(:,2) = brgb(regionIdx).SpatialReferencing(1).YWorldLimits(2)-(testCoords{k}(:,2)-brgb(regionIdx).SpatialReferencing(1).YWorldLimits(1));
    end
    clear testStruct testStructs

    regionIdx = 1;
    for k = 1:numTest
        if k > numTestRegionsCumulative(regionIdx)
            regionIdx = regionIdx + 1;
        end
        
        coords = testCoords{k};
        regionImg = getRegion(brgb(regionIdx),1,[min(coords(:,1)) min(coords(:,2))],[max(coords(:,1)) max(coords(:,2))]);
        imgFilename = fullfile("test_data", testID{k}+".png" );
        imwrite(regionImg,imgFilename);
    end
    save(fullfile("test_data","test_data"),"testID","testCoords")
end

%% EXPLORING THE DATA
imds = imageDatastore("training_data","IncludeSubfolders",true, "FileExtensions",".png","LabelSource","foldernames")
labelInfo = countEachLabel(imds)

%% TRAINING A NEURAL NETWORK USING TRANSFER LEARNING
% _NOTE: You will first have to download the Deep Learning Toolbox Model for ResNet-18 Network support package._

net = resnet18;

% To retrain ResNet-18 to classify new images, replace the last fully connected layer and the final classification layer of the network. 
% In ResNet-18, these layers have the names |'fc1000'| and |'ClassificationLayer_predictions'|, respectively. 
% Set the new fully connected layer to have the same size as the number of classes in the new data set. 
% To learn faster in the new layers than in the transferred layers, increase the learning rate factors of the fully connected layer using the |'WeightLearnRateFactor'| and |'BiasLearnRateFactor'| properties.

numClasses = numel(categories(imds.Labels));
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
analyzeNetwork(lgraph);

% Set up training options

% Configure the image datastore to use the neural network's required input image size. 
% To do this, we are registering a custom function called <internal:959032C0 |readAndResize|> (which you can find at the end of this script) and setting it as the |ReadFcn| in the datastore.

inputSize = net.Layers(1).InputSize;
imds.ReadFcn = @(im)readAndResize(im,inputSize); % Refers to a helper function at the end of this script

% Split the training data into training and validation sets. 
% Note that this is randomly selecting a split, but you may want to look into the <https://www.mathworks.com/help/releases/R2019b/matlab/ref/matlab.io.datastore.imagedatastore.spliteachlabel.html |splitEachLabel|> function for other options to make sure the classes are balanced

[imdsTrain,imdsVal] = splitEachLabel(imds,0.7,"randomized");
% Specify the training options, including mini-batch size and validation data. 
% Set |InitialLearnRate| to a small value to slow down learning in the transferred layers.
% In the previous step, you increased the learning rate factors for the fully connected layer to speed up learning in the new final layers. 
% This combination of learning rate settings results in fast learning only in the new layers and slower learning in the other layers.

% You can work with different options to improve the training. Check out the https://www.mathworks.com/help/releases/R2019b/deeplearning/ref/trainingoptions.html documentation for trainingOptions> to learn more.

options = trainingOptions('sgdm', 'MiniBatchSize', 32, 'MaxEpochs', 5, 'InitialLearnRate', 1e-4, 'Shuffle','every-epoch', 'ValidationData',imdsVal, 'ValidationFrequency',floor(numel(imdsTrain.Files)/(32*2)), 'Verbose' ,false , 'Plots', 'training-progress')
% Train the network

% Here you will use the image datastores, neural network layer graph, and training options to train your model.
% Note that training will take a long time using a CPU. 
% However, MATLAB will automatically detect if you have a <https://www.mathworks.com/help/releases/R2019b/parallel-computing/gpu-support-by-release.html supported GPU> to help you accelerate training.
% Set the |doTraining| flag below to |false| to load a presaved network.

doTraining = true;
if doTraining
    netTransfer = trainNetwork(imdsTrain,lgraph,options);
else
    load resnet_presaved.mat
end

% PREDICTING & SUBMITTING

% Predict on the Test Set
% Once we have our trained network, we can perform predictions on our test set. 
% To do so, first we will create an image datastore for the test set.

imdsTest = imageDatastore("test_data","FileExtensions",".png");
imdsTest.ReadFcn = @(im)readAndResize(im,inputSize)

% Next we predict labels (|testMaterial)| and scores (|testScores|) using the trained network 
% NOTE: This will take some time, but just as with training the network, MATLAB will determine whether you have a supported GPU and significantly speed up this process.

[testMaterial,testScores] = classify(netTransfer,imdsTest)
% The following code will display the predicted materials for a few test images. 
% You need to change the |display| flag to |true| to execute the below code. 

%figure
%displayIndices = randi(numTest,4,1);
%for k = 1:numel(displayIndices)
%    testImg = readimage(imdsTest,displayIndices(k));
%    subplot(2,2,k)
%    imshow(testImg);
%    title(string(testMaterial(displayIndices(k))),"Interpreter","none")
%end

% Save Submission to File
%% 

testResults = table(testID,testScores(:,1),testScores(:,2), testScores(:,3),testScores(:,4),testScores(:,5), 'VariableNames',['id';categories(testMaterial)])
writetable(testResults,'testResults.csv');
%% 

function [rgb, m] = separateChannels(rgbm)
rgb = rgbm(:,:,1:3);
m = logical(rgbm(:,:,4));
end

function im = readAndResize(filename,sz)
im = imresize( imread(filename), sz(1:2) );
end