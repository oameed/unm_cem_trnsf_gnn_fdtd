%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO %%%
%%% COMPUTATIONAL EM LAB     %%%
%%% DEEP LEARNING PROJECT    %%%
%%% GIF GENERATOR            %%%
%%% by: OAMEED NOAKOASTEEN   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS: NET                   : NETWORK VERSION
% INPUT PARAMETERS: CH                    : NUMBER OF CHANNELS
% INPUT PARAMETERS: OPT                   : '3-FIELD-COMPONENTS': 'Power'/'Emag'/'Ex'/'Ey'/'Hz'
% INPUT PARAMETERS: OPT                   : '2-FIELD-COMPONENTS': 'Emag'/'Ex'/'Ey'
% INPUT PARAMETERS: OPT                   : '1-FIELD-COMPONENTS': 'Hz'
% INPUT PARAMETERS: FILEDCOLORSCALEFACTOR : 
% INPUT PARAMETERS: ERRORCOLORSCALEFACTOR : COLOR SCALE (CAXIS) FOR ERROR PLOT
% INPUT PARAMETERS: PASF                  : PRED AXIS SCALE FACTOR    
% hdfdir                                  : HDF FILE DIRECTORIES
% permorder                               : PERMUTE ORDER FOR HDF DATA
% field_color_scale_factor                : COLOR SCALE FACTOR (CAXIS) FOR FIELD PLOT
% error_color_scale_factor                : COLOR SCALE FACTOR (CAXIS) FOR ERROR PLOT
% PAUSE                                   : PASUE TIME BETWEEN GIF FRAMES

function graphics(NET,CH,OPT,FILEDCOLORSCALEFACTOR,ERRORCOLORSCALEFACTOR,PASF)
close all
clc
hdfdir                     =["vid_true",'vid_pred','bnd']                                              ; 
permorder                  =[4,2,3,1]                                                                  ; 
field_color_scale_factor   =FILEDCOLORSCALEFACTOR                                                      ; 
error_color_scale_factor   =ERRORCOLORSCALEFACTOR                                                      ; 
PAUSE                      =0.1                                                                        ; 

[readdatapath,savedatapath]=getDATAPATH(NET)                                                           ;
filenames                  =getFILENAMES(readdatapath)                                                 ;
error_scales               =get_error_scales(NET)                                                      ; % USE ERROR SCALES FROM THE ERD ARCHITECTURE.
                                                                                                         % THESE ARE COMPUTED USING THE "get_ERD_Predictions_color_scale"
                                                                                                         % WHICH IS IN '..\..\lib'. THIS FUNCTION MUST
                                                                                                         % BE PLACED IN THE 'run\fdtd' DIRECTORY OF THE
                                                                                                         % ERD PROJECT AND USED AS: 
                                                                                                         % "get_ERD_Predictions_color_scale('type_2','v23','CONTVID')"

for i=1:size(filenames,2)
    savefilename=fullfile(savedatapath,getSAVENAME(filenames(i)));
    vid_true    =permute(h5read(fullfile(readdatapath,filenames(i)),strcat('/',hdfdir(1))),permorder)  ;
    vid_pred    =permute(h5read(fullfile(readdatapath,filenames(i)),strcat('/',hdfdir(2))),permorder)  ;
    bnd         =        h5read(fullfile(readdatapath,filenames(i)),strcat('/',hdfdir(3)))             ;
    data_true   =getPLOTDATA(vid_true,CH,OPT)                                                          ;
    data_pred   =getPLOTDATA(vid_pred,CH,OPT)                                                          ;
    field_scale =field_color_scale_factor.*getFIELDSCALE(vid_true,CH)                                  ;
    %error_scale =error_color_scale_factor.*(getMAXNORM(vid_true-vid_pred))                             ;
    error_scale =error_color_scale_factor.*error_scales(i)                                             ; 
    GENGIF(bnd,data_true,data_pred,OPT,field_scale,error_scale,error_color_scale_factor,PAUSE,savefilename,PASF)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FUNCTION DEFINITIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function scales=get_error_scales(NET)
        if strcmp(NET,'v11')
            TYPE='type1'                         ;
            NAME='color_scale_type_1_CONTVID.csv';
        else
            if strcmp(NET,'v12')
                TYPE='type2'                         ;
                NAME='color_scale_type_2_CONTVID.csv';
            else
                if strcmp(NET,'v13')                    
                    TYPE='type3'                         ;
                    NAME='color_scale_type_3_CONTVID.csv';
                else
                    if strcmp(NET,'v21')
                        TYPE='type1'                         ;
                        NAME='color_scale_type_1_TESTVID.csv';
                    else
                        if strcmp(NET,'v22')
                            TYPE='type2'                         ;
                            NAME='color_scale_type_2_TESTVID.csv';
                        else
                            if strcmp(NET,'v23')
                                TYPE='type3';
                                NAME='color_scale_type_3_TESTVID.csv';
                            end
                        end
                    end
                end
            end
        end
        scales=load(fullfile('..','..','data',TYPE,'info',NAME));
    end

    function [RDATAPATH,SDATAPATH]=getDATAPATH(NET)
        netpath  =fullfile('..','..','networks',NET             )                                      ;
        RDATAPATH=fullfile(netpath,'predictions','hdf'          )                                      ;
        SDATAPATH=fullfile(netpath,'predictions'                )                                      ;
    end

    function FILENAMES=getFILENAMES(READDATAPATH)
        dirlist  =dir(READDATAPATH)                                                                     ;
        FILENAMES=string({dirlist(4:end).name})                                                         ;
    end

    function data=getPLOTDATA(DATA,CH,OPT)
        if CH==1
            data=DATA                                                                                   ;
        else
            if CH==2
                if strcmp(OPT,'Emag')
                    data=sqrt(DATA(:,:,:,1).^2+DATA(:,:,:,2).^2)                                        ;
                else
                    if strcmp(OPT,'Ex')
                        data=DATA(:,:,:,1)                                                              ;
                    else
                        if strcmp(OPT,'Ey')
                            data=DATA(:,:,:,2)                                                          ;
                        end
                    end
                end
            else
                if CH==3
                    if strcmp(OPT,'Power')
                        data=(1/2).*abs(DATA(:,:,:,3)).*sqrt(DATA(:,:,:,1).^2+DATA(:,:,:,2).^2)         ;
                    else
                        if strcmp(OPT,'Emag')
                            data=sqrt(DATA(:,:,:,1).^2+DATA(:,:,:,2).^2)                                ;
                        else
                            if strcmp(OPT,'Ex')
                                data=DATA(:,:,:,1)                                                      ;
                            else
                                if strcmp(OPT,'Ey')
                                    data=DATA(:,:,:,2)                                                  ;
                                else
                                    if strcmp(OPT,'Hz')
                                        data=DATA(:,:,:,3)                                              ;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
           
    function temp=getSAVENAME(FILENAME)
       temp=split(FILENAME,'.')                                                                         ;
       temp=temp(1)                                                                                     ;
       temp=strcat(temp,'.gif')                                                                         ;
    end
    
    function scale=getFIELDSCALE(DATA,CH)
        if CH==1
            scale=max(abs(DATA(:)))                                                                     ;
        else
            if CH==2
                data =sqrt((DATA(:,:,:,1)).^2+(DATA(:,:,:,2)).^2)                                       ;
                scale=max(abs(data(:)))                                                                 ;
            else
                if CH==3
                    data =0.5.*(DATA(:,:,:,3)).*sqrt((DATA(:,:,:,1)).^2+(DATA(:,:,:,2)).^2)             ;
                    scale=max(abs(data(:)))                                                             ;
                end
            end
        end     
    end
  
    function scale=getMAXNORM(DATA)
        fnorm=[]                                                                                        ;
        for J=1:size(DATA,4)
            data     =squeeze(DATA(:,:,:,J))                                                            ;
            for I=1:size(DATA,1)
                fnorm=[fnorm,norm(squeeze(data(I,:,:)),'fro')]                                          ;
            end
        end
        scale=max(fnorm)                                                                                ;
    end

    function plotter(BND,TRUE,PRED,OPT,INDEX,FIELD_SCALE,ERROR_SCALE,ERROR_SCALE_FACTOR,PASF)
        CAXISMAX=FIELD_SCALE;
        subplot(1,3,1)
        surf(TRUE)
        title(['TEz ',' TURE ',OPT,' ',num2str(INDEX)])
        set(gca,'xticklabel',[],'yticklabel',[])
        colormap default
        shading  interp
        axis     equal
        caxis    manual
        if strcmp(OPT,'Power')
            xlabel(['\times',' ^{\eta}/_{\it {Scale Factor}^2} '])
            caxis( [0 CAXISMAX                                  ])
        else
            if strcmp(OPT,'Emag')
                xlabel(['\times',' ^{1}/_{\it Scale Factor} '])
                caxis( [0 CAXISMAX                           ])
            else
                if strcmp(OPT,'Ex')
                    xlabel(['\times',' ^{1}/_{\it Scale Factor} '])
                    caxis( [-CAXISMAX CAXISMAX                   ])
                else
                    if strcmp(OPT,'Ey')
                        xlabel(['\times',' ^{1}/_{\it Scale Factor} '])
                        caxis( [-CAXISMAX CAXISMAX                   ])
                    else
                        if strcmp(OPT,'Hz')
                            xlabel(['\times',' ^{\eta}/_{\it Scale Factor} '])
                            caxis( [-CAXISMAX CAXISMAX                      ])
                        end
                    end
                end
            end
        end
        view(2)
        subplot(1,3,2)
        surf(PRED)
        title(['TEz ',' PRED ',OPT,' ',num2str(INDEX)])
        set(gca,'xticklabel',[],'yticklabel',[])
        colormap default
        shading  interp
        axis     equal
        caxis    manual
        if strcmp(OPT,'Power')
            xlabel(['\times',' ^{\eta}/_{\it {Scale Factor}^2} '            ])
            caxis( [0 PASF*CAXISMAX                                         ])
        else
            if strcmp(OPT,'Emag')
                xlabel(['\times',' ^{1}/_{\it Scale Factor} '               ])
                caxis( [0 PASF*CAXISMAX                                     ])
            else
                if strcmp(OPT,'Ex')
                    xlabel(['\times',' ^{1}/_{\it Scale Factor} '           ])
                    caxis( [-PASF*CAXISMAX PASF*CAXISMAX                    ])
                else
                    if strcmp(OPT,'Ey')
                        xlabel(['\times',' ^{1}/_{\it Scale Factor} '       ])
                        caxis( [-PASF*CAXISMAX PASF*CAXISMAX                ])
                    else
                        if strcmp(OPT,'Hz')
                            xlabel(['\times',' ^{\eta}/_{\it Scale Factor} '])
                            caxis( [-PASF*CAXISMAX PASF*CAXISMAX            ])
                        end
                    end
                end
            end
        end
        view(2)
        subplot(1,3,3)
        surf(abs(PRED-TRUE).^2)            % !!!
        %title(['Magnitude of Error ',num2str(INDEX)])
        title(['|TRUE-PRED|^{2}, Frame ',num2str(INDEX)])
        xlabel({'Color Scale:';[num2str(ERROR_SCALE_FACTOR*100),'%',' \times','FrobNorm_{max}']})
        set(gca,'xticklabel',[],'yticklabel',[])        
        colormap default
        shading  interp
        axis     equal        
        caxis    manual
        caxis([0 ERROR_SCALE])
        originalSize=get(gca, 'Position'); % !!!
        colorbar                           % !!!
        set(gca,'Position',originalSize)   % !!!
        view(2)
    end

    function GENGIF(BND,TRUE,PRED,OPT,FIELD_SCALE,ERROR_SCALE,ERROR_SCALE_FACTOR,PAUSE,NAME,PASF)
        MATLAB_FIGURE_SIZE_SCALE=1.25                                                                   ;
        close all
        h  =figure('Position',[520  378 MATLAB_FIGURE_SIZE_SCALE*560  MATLAB_FIGURE_SIZE_SCALE*420])    ;
        sgt=sgtitle({'University of New Mexico','Computational EM Lab'});
        sgt.FontSize=10                                                                                 ;
        for index=1:floor(0.75*size(TRUE,1)) % EXCLUDE EMPTY FRAMES
            plotter(BND,squeeze(TRUE(index,:,:)),squeeze(PRED(index,:,:)),OPT,index,FIELD_SCALE,ERROR_SCALE,ERROR_SCALE_FACTOR,PASF)
            [imind,cm]=rgb2ind(frame2im(getframe(h)),256)                                               ;
            if index==1
                imwrite(imind,cm,NAME,'gif','Loopcount',inf     ,'DelayTime',PAUSE)
            else
                imwrite(imind,cm,NAME,'gif','WriteMode','append','DelayTime',PAUSE)
            end            
        end
    end
end
