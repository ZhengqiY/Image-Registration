close all
clear
clc

im1 = imread('overhead_1.jpg') ;
im2 = imread('overhead_2.jpg') ;
% make single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% make grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1); else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

figure, imshow(im1g);
[f1,d1] = vl_sift(im1g) ;
fprintf('Number of frames (features) detected: %d\n', size(f1,2));
h = vl_plotframe(f1);
set(h,'color','y','linewidth',1);

figure, imshow(im2g);
[f2,d2] = vl_sift(im2g) ;
fprintf('Number of frames (features) detected: %d\n', size(f2,2));
h = vl_plotframe(f2);
set(h,'color','y','linewidth',1);

[matches, scores] = vl_ubcmatch(d1,d2) ;
fprintf('Number of matching frames (features): %d\n', size(matches,2));
indices1 = matches(1,:); % Get matching features
f1match = f1(:,indices1);
d1match = d1(:,indices1);
indices2 = matches(2,:);
f2match = f2(:,indices2);
d2match = d2(:,indices2);

figure, imshow([im1g,im2g]);
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

% numMatches = size(matches,2);
% 
 pts1 = f1match([1,2],[2,4]);
 pts2 = f2match([1,2],[2,4]);
% 
 H = vgg_H_from_x_lin(pts1, pts2);

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
mosaic = (im1_ + im2_) ./ mass ;

figure(4) ; clf ;
imagesc(mosaic) ; axis image off ;
title('Mosaic') ;
