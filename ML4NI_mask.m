function [m_img, m_hdr, m_ind] = ML4NI_mask(Y, H)
% _
% Create Mask for Second-Level Analysis [1]
% FORMAT [m_img, m_hdr, m_ind] = MS_create_mask(Y, H)
% 
%     Y     - an N x V matrix of second-level data (N: number of subjects)
%     H     - a structure with exemplary header information for these data
% 
%     m_img - a 1 x V mask vector (V: number of voxels)
%     m_hdr - a structure with mask header information
%     m_ind - a 1 x v vector indicating in-mask voxels
% 
% FORMAT [m_img, m_hdr, m_ind] = ML4NI_mask(Y, H) creates a mask
% image for the data Y, such that all columns of this data matrix that
% have no NaN value are inside the mask [1], and a mask header that fits
% the input H. Then, it returns mask image and mask header along with
% indices of all in-mask voxels.
% 
% References:
% [1] JoramSoch (2017): "Create Mask for Second-Level Analysis"; URL:
%     https://github.com/JoramSoch/MACS/blob/master/MS_create_mask.m
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 09/12/2014, 13:40
%  Last edit: 27/07/2021, 09:03


% Init progress bar
%-------------------------------------------------------------------------%
Finter = spm('FigName','ML4NI_mask: create');

% Create mask image
%-------------------------------------------------------------------------%
m_img = double(~sign(sum(isnan(Y),1)));
m_ind = find(m_img~=0);
m_hdr = H;

% Create mask header
%-------------------------------------------------------------------------%
m_hdr.fname   = 'mask.nii';
m_hdr.dt      = [spm_type('uint8') spm_platform('bigend')];
m_hdr.descrip = 'ML4NI_mask: second-level mask';