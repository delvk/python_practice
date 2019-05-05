%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pkg load image

clc; clear all;

// Input path
path = ['../data/test_data/images/'];
gt_path = ['../data/test_data/ground_truth/'];
// Output path
gt_path_csv = ['../data/test_data/ground_truth_csv/'];

mkdir(gt_path_csv )

for i = 1:316    
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end     
    annPoints =  image_info{1}.location;   
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    csvwrite([gt_path_csv ,'IMG_',num2str(i) '.csv'], im_density); 
    if(i==316) fprintf('Finished')      
end

