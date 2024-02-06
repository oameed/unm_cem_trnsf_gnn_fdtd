%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO               %%%
%%% COMPUTATIONAL EM LAB                   %%%
%%% EMULATING FDTD USING TRANSFORMER & GNN %%%
%%% FDTD TEz DATASET GENERATOR             %%%
%%% by: OAMEED NOAKOASTEEN                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TYPE         : 'type1','type2','type3'
% SPECLIST     : true/false
% SPECPARAMS(1): TOTAL NUMBER OF SIMULATIONS
% SPECPARAMS(2): RANGE OF VALUES FOR ANGLE OF PROPAGATION OF THE TFSF SOURCE
% SPECPARAMS(3): RANGE OF VALUES FOR POINT SOURCE LOCATIONS
% SPECPARAMS(4): RANGE OF VALUES FOR OBJECT       LOCATIONS
% SPECPARAMS(5): RANGE OF VALUES FOR OBJECT SIZES
% PATH(1)      : INFO  PATH
% PATH(2)      : WRITE PATH
% SPEC(1)      : ANGLE OF PROPAGATION OF THE TFSF         SOURCE
% SPEC(2)      : X-LOCATION           OF THE FIRST  POINT SOURCE 
% SPEC(3)      : Y-LOCATION           OF THE FIRST  POINT SOURCE
% SPEC(4)      : X-LOCATION           OF THE SECOND POINT SOURCE
% SPEC(5)      : Y-LOCATION           OF THE SECOND POINT SOURCE
% SPEC(6): L   : LENGTH OF DOMAIN
% SPEC(7)      : X-LOCATION  OF THE FIRST  OBJECT
% SPEC(8)      : Y-LOCATION  OF THE FIRST  OBJECT
% SPEC(9)      : X-LOCATION  OF THE SECOND OBJECT
% SPEC(10)     : Y-LOCATION  OF THE SECOND OBJECT
% SPEC(11)     : OBJECT TYPE AND NUMBER
%                 '1': SQUARE
%                 '2': CIRCLE
%                 '3': MIX OF BOTH
% SPEC(12)     : SIZE FACTOR OF THE FIRST  OBJECT
% SPEC(13)     : SIZE FACTOR OF THE SECOND OBJECT
% TIME         : TIMING FOR EACH ITERATION OF FDTD UPDATE FOR EACH SIMULATION

function generator(TYPE,SPECLIST,SPECPARAMS)
close all
clc
addpath(         fullfile('..','..','data','solver' ,'fdtd'      ))
PATH        =  { fullfile('..','..','data',TYPE     ,'info'      ),...
                 fullfile('..','..','data',TYPE     ,'hdf5','raw'),...
                 fullfile('..','..','data',TYPE     ,'gif' ,'raw')}                                               ;
specfilename=    fullfile(PATH{1}  ,strcat('simspec','.csv'      ))                                               ;
timefilename=    fullfile(PATH{1}  ,strcat('simtime','.csv'      ))                                               ;
TIME        = []                                                                                                  ;
PAUSE       = 0.1                                                                                                 ;

if SPECLIST
    spec    =get_specs(TYPE,SPECPARAMS)                                                                           ;
    writematrix(spec,specfilename)
    disp([' GENERATED SPEC FILE FOR ',' ',TYPE])
else
    disp([' READING   SPEC FILE FOR ',' ',TYPE])
    spec    =load(specfilename)                                                                                   ;    
end

for index=1:size(spec,1)
    disp([' GENERATING SIMULATION ',' ',num2str(index)])
    savefilename=fullfile(PATH{2},strcat('simulation','_',num2str(index),'.h5'))                                  ;
    time=FDTD2DTEz(TYPE,spec(index,:),savefilename)                                                               ;
    TIME=[TIME;time]                                                                                              ;
end
writematrix(TIME,timefilename)

disp(' GENERATING GIFs ')

list=getFILENAMES(PATH{2})                                                                                        ;
for index=1:size(list,1)
    [Ex,Ey,Hz]=readDATA(PATH{2},list{index})                                                                      ;
    genGIF({Ex,Ey,Hz},{list{index},PATH{3},PAUSE})
end

disp(' FINISHED ')

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%% FUNCTION DEFINITIONS %%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    function offset=get_offset(L,LAMMIN,FRAC)        
        offset=FRAC*LAMMIN/L;        
    end
 
    function spec=get_specs(TYPE,SPECPARAMS)        
        spec             =[]                                                                                      ;
        offset           =get_offset(1.5,0.149896,0.6)                                                            ;
        numtotal         =SPECPARAMS{1}                                                                           ;
        shape            =[1,2,3]                                                                                 ;        
        if strcmp(TYPE,'type1')
            source_tfsf_theta=SPECPARAMS{2}                                                                       ;
            obj_1_pos        =SPECPARAMS{4}                                                                       ;
            obj_2_pos        =SPECPARAMS{4}                                                                       ;
            obj_size         =SPECPARAMS{5}                                                                       ;
            for index1=1:size(source_tfsf_theta,2)
                for index2=1:size(obj_1_pos,2)
                    for index3=1:size(obj_1_pos,2)
                        for index4=1:size(obj_2_pos,2)
                            for index5=1:size(obj_2_pos,2)
                                for index6=1:size(shape,2)
                                    for index7=1:size(obj_size,2)
                                        for index8=1:size(obj_size,2)
                                            x   =[source_tfsf_theta(index1),...
                                                  obj_1_pos(index2)        ,...
                                                  obj_1_pos(index3)        ,...
                                                  obj_2_pos(index4)        ,...
                                                  obj_2_pos(index5)        ,...
                                                  shape(index6)            ,...
                                                  obj_size(index7)         ,...
                                                  obj_size(index8)         ]                                      ;
                                            x   =[x(1),0,0,0,0,1.5,x(2),x(3),x(4),x(5),x(6),x(7),x(8)]            ;
                                            spec=[spec;x]                                                         ;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            spec=spec(randperm(size(spec,1),numtotal),:)                                                          ;
        else
            if strcmp(TYPE,'type2')
                source_point_1_pos=SPECPARAMS{3}-offset                                                           ;
                obj_1_pos         =SPECPARAMS{4}                                                                  ;
                obj_2_pos         =SPECPARAMS{4}                                                                  ;
                obj_size          =SPECPARAMS{5}                                                                  ;
                for index1=1:size(source_point_1_pos,2)
                    for index2=1:size(source_point_1_pos,2)
                        for index3=1:size(obj_1_pos,2)
                            for index4=1:size(obj_1_pos,2)
                                for index5=1:size(obj_2_pos,2)
                                    for index6=1:size(obj_2_pos,2)
                                        for index7=1:size(shape,2)
                                            for index8=1:size(obj_size,2)
                                                for index9=1:size(obj_size,2)
                                                    x   =[source_point_1_pos(index1),...
                                                          source_point_1_pos(index2),...
                                                          obj_1_pos(index3)         ,...
                                                          obj_1_pos(index4)         ,...
                                                          obj_2_pos(index5)         ,...
                                                          obj_2_pos(index6)         ,...
                                                          shape(index7)             ,...
                                                          obj_size(index8)          ,...
                                                          obj_size(index9)          ]                             ;
                                                    x   =[0,x(1),x(2),0,0,1.5,x(3),x(4),x(5),x(6),x(7),x(8),x(9)] ;
                                                    spec=[spec;x]                                                 ;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                spec=spec(randperm(size(spec,1),numtotal),:)                                                      ;
            else
                if strcmp(TYPE,'type3')
                    source_point_1_pos=SPECPARAMS{3}                                                              ;
                    source_point_2_pos=SPECPARAMS{3}+offset                                                       ;
                    for index1=1:size(source_point_1_pos,2)
                        for index2=1:size(source_point_1_pos,2)
                            for index3=1:size(source_point_2_pos,2)
                                for index4=1:size(source_point_2_pos,2)
                                    x   =[source_point_1_pos(index1),...
                                          source_point_1_pos(index2),...
                                          source_point_2_pos(index3),...
                                          source_point_2_pos(index4)]                                             ;
                                    x   =[0,x(1),x(2),x(3),x(4),1.5,0.35,0.35,0,0,2,0.70,0]                       ;
                                    spec=[spec;x]                                                                 ;
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    function list=getFILENAMES(PATH)
        list=dir(PATH)                                                                                            ;
        list=string({list(4:end).name})'                                                                          ;
    end

    function [ex,ey,hz]=readDATA(READPATH,NAME)
        ex=h5read(fullfile(READPATH,NAME),strcat('/','Ex_FIELD'))                                               ;
        ey=h5read(fullfile(READPATH,NAME),strcat('/','Ey_FIELD'))                                               ;
        hz=h5read(fullfile(READPATH,NAME),strcat('/','Hz_FIELD'))                                               ;
    end

    function wname=getWNAME(WNAME)
        splits=split(WNAME,'.')                                                                                   ;
        wname =strcat(splits{1},'.gif')                                                                           ;
    end

    function plotfunc(DATA,INDEX)
        surf(DATA)
        title({'University of New Mexico','Computational EM Lab'},'FontSize',10)
        xlabel(num2str(INDEX))
        set(gca,'xticklabel',[],'yticklabel',[])
        axis     equal
        colormap default
        shading  interp
        view(2)
    end

    function genGIF(DATA,PARAM)
        close all
        h        =figure                                                                                          ;
        ex       =DATA{1}                                                                                         ;
        ey       =DATA{2}                                                                                         ;
        hz       =DATA{3}                                                                                         ;
        wNAME    =getWNAME(PARAM{1})                                                                              ;
        wPATH    =PARAM{2}                                                                                        ;
        pause    =PARAM{3}                                                                                        ;
        power    =0.5.*abs(hz).*sqrt(ex.^2+ey.^2)                                                                 ;        
        wFILENAME=fullfile(wPATH,wNAME)                                                                           ;
        for INDEX=1:size(ex,3)
            plotfunc(power(:,:,INDEX),INDEX)
            [imind,cm]=rgb2ind(frame2im(getframe(h)),256)                                                         ;
            if INDEX==1
                imwrite(imind,cm,wFILENAME,'gif','Loopcount',inf     ,'DelayTime',pause)
            else
                imwrite(imind,cm,wFILENAME,'gif','WriteMode','append','DelayTime',pause)
            end
        end
    end

end

