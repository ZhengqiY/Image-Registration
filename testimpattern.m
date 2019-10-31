close all
clear
clc
% --------------------------------------------------------------------
% Load reference image
% --------------------------------------------------------------------

im1 = imread('overhead_1.jpg') ;
% Make it single precision
im1 = im2single(im1) ;
% Make it grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1); else im1g = im1 ; end
figure(1), imshow(im1g);

% --------------------------------------------------------------------
% Compute the SIFT frames (keypoints) and descriptors for the Ref Im                                                         
% --------------------------------------------------------------------

[f1,d1] = vl_sift(im1g) ;
fprintf('Number of frames (features) detected: %d\n', size(f1,2));
h = vl_plotframe(f1);
set(h,'color','y','linewidth',1);

% --------------------------------------------------------------------
% Load To be registered image
% --------------------------------------------------------------------

im2 = imread('overhead_2.jpg') ;
% Make it single precision
im2 = im2single(im2) ;
% Make it grayscale
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end
figure(2), imshow(im2g);

% --------------------------------------------------------------------
% Compute the SIFT frames (keypoints) and descriptors for the Reg Im                                                         
% --------------------------------------------------------------------

[f2,d2] = vl_sift(im2g) ;
fprintf('Number of frames (features) detected: %d\n', size(f2,2));
h = vl_plotframe(f2);
set(h,'color','g','linewidth',1);

% --------------------------------------------------------------------
% Extract and Match the descriptors                                                        
% --------------------------------------------------------------------

[matches, scores] = vl_ubcmatch(d1,d2) ;
fprintf('Number of matching frames (features): %d\n', size(matches,2));
indices1 = matches(1,:); % Get matching features
f1match = f1(:,indices1);
d1match = d1(:,indices1);
indices2 = matches(2,:);
f2match = f2(:,indices2);
d2match = d2(:,indices2);

figure(3), imshow([im1g,im2g]);
o = size(im1g,2) ;
line([f1match(1,:);f2match(1,:)+o],[f1match(2,:);f2match(2,:)]) ;
for i=1:size(f1match,2)
 x = f1match(1,i);
 y = f1match(2,i);
 text(x,y,sprintf('%d',i), 'Color', 'r');
end

for i=1:size(f2match,2)
 x = f2match(1,i);
 y = f2match(2,i);
 text(x+o,y,sprintf('%d',i), 'Color', 'r');
end

% --------------------------------------------------------------------
% Pick random N points for DLT/Normalized DLT/ DLT + RANSAC to estimate the
% homography
% --------------------------------------------------------------------

pts1 = f1match([1,2,3],[2,3,4,5]);
pts2 = f2match([1,2,3],[2,3,4,5]);
% DLT Algorithm
H = vgg_H_from_x_lin(pts1, pts2);
% Normalized DLT Algorithm
H_norm = vgg_H_from_x_lin(normalise2dpts(pts1),normalise2dpts(pts2));

%H2 = ransacfithomography(pts1, pts2, 0.002);

% --------------------------------------------------------------------
% Image Warping under DLT
% --------------------------------------------------------------------

box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
box2_ = inv(H) * box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
ur = min([1 box2_(1,:)]):max([size(im1,2) box2_(1,:)]) ;
vr = min([1 box2_(2,:)]):max([size(im1,1) box2_(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
im1_ = vl_imwbackward(im2double(im1),u,v) ;

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
im2_ = vl_imwbackward(im2double(im2),u_,v_) ;

mass = ~isnan(im1_) + ~isnan(im2_) ;
im1_(isnan(im1_)) = 0 ;
im2_(isnan(im2_)) = 0 ;
mosaic_DLT = (im1_ + im2_) ./ mass ;

figure(4) ; clf ;
imagesc(mosaic_DLT) ; axis image off ;
title('Mosaic DLT') ;

% --------------------------------------------------------------------
% Image Warping under normalized DLT
% --------------------------------------------------------------------

box3 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
box3_ = inv(H_norm) * box3 ;
box3_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box3_(2,:) = box3_(2,:) ./ box3_(3,:) ;
ur2 = min([1 box3_(1,:)]):max([size(im1,2) box3_(1,:)]) ;
vr2 = min([1 box3_(2,:)]):max([size(im1,1) box3_(2,:)]) ;

[u2,v2] = meshgrid(ur2,vr2) ;
im1_norm = vl_imwbackward(im2double(im1),u2,v2) ;

z2_ = H_norm(3,1) * u2 + H_norm(3,2) * v2 + H_norm(3,3) ;
u2_ = (H_norm(1,1) * u2 + H_norm(1,2) * v2 + H_norm(1,3)) ./ z2_ ;
v2_ = (H_norm(2,1) * u2 + H_norm(2,2) * v2 + H_norm(2,3)) ./ z2_ ;
im2_norm = vl_imwbackward(im2double(im2),u2_,v2_) ;

mass2 = ~isnan(im1_norm) + ~isnan(im2_norm) ;
im1_norm(isnan(im1_norm)) = 0 ;
im2_norm(isnan(im2_norm)) = 0 ;
mosaic_normDLT = (im1_norm + im2_norm) ./ mass2 ;

figure(5) ; clf ;
imagesc(mosaic_normDLT) ; axis image off ;
title('Mosaic norm DLT') ;
