function [c1,c2]=image_processing(in_dir,filename)

%filename="gecko2";
threshold=0.9;
swap_image=true;
pixel_rate=3;
degree=5;
plot_on=false;

%n_dir="./data/mesh_images";
out_dir="./mesh_points/";
Image = imread(strcat(in_dir,filename,'.png'));

I     = im2bw(Image,threshold);
if swap_image
    I=~I;
end
Ifill = imfill(I,'holes');
B = bwboundaries(Ifill);
for k = 1
    b = B{k};
    if plot_on
        hold on
        subplot(121)
        title("Original Image')")
        plot(b(:,1),b(:,2),'r','linewidth',2);
    end
end


c1  = downsample(b(:,1),pixel_rate);
c2  = downsample(b(:,2),pixel_rate);
if plot_on
    subplot(122)
    title('Filtered Grid')
    plot(c1,c2)
end
disp(strcat(out_dir,filename,'.mat'))
save (strcat(out_dir,filename,'.mat'), 'c1', 'c2')


% fileID = fopen(strcat(out_dir,filename,'_discrete_mesh_generator.py'),'w');
% 
% fprintf(fileID,"import gmsh \n");
% fprintf(fileID,"import sys \n \n");
% fprintf(fileID,"gmsh.initialize() \n");
% fprintf(fileID,"gmsh.model.add(""%s"") \n \n",filename);
% fprintf(fileID,"gmsh.model.addDiscreteEntity(1, 100) \n");
% fprintf(fileID,"flat_pts=[] \n\n");
% for i=1:length(c1)
%     fprintf(fileID,"flat_pts.append(%d) \n",c1(i));
%     fprintf(fileID,"flat_pts.append(%d) \n",c2(i));
%     fprintf(fileID,"flat_pts.append(%d) \n",0);
% end
% fprintf(fileID,"\ngmsh.model.mesh.addNodes(1, 100, range(1, %d + 1), flat_pts) \n",length(c1));
% fprintf(fileID,"n = [item for sublist in [[i, i + 1] for i in range(1, %d + 1)] for item in sublist] \n",length(c1));
% fprintf(fileID,"n[-1] = 1 \n");
% fprintf(fileID,"gmsh.model.mesh.addElements(1, 100, [1], [range(1, %d + 1)], [n]) \n",length(c1));
% fprintf(fileID,"gmsh.model.geo.addCurveLoop([100], 101) \ngmsh.model.geo.addPlaneSurface([101], 102) \ngmsh.model.geo.synchronize() \n");
% fprintf(fileID,"gmsh.model.mesh.generate(2) \n");
% fprintf(fileID,"gmsh.write(""%s"") \n",strcat(my_dir,mesh_dir,filename,".msh"));
% fprintf(fileID,"if '-nopopup' not in sys.argv: \n");
% fprintf(fileID,"\tgmsh.fltk.run() \n");
% fprintf(fileID,"gmsh.finalize()");


disp("File Made!")
end

