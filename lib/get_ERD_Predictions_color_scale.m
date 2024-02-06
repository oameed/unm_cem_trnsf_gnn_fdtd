function get_ERD_Predictions_color_scale(DATASET,NET,TYPE)
close all
clc

readdatapath=getDATAPATH(NET,TYPE)                                       ;
savefilename=strcat('color_scale','_',DATASET,'_',TYPE,'.csv')           ;
filenames   =getFILENAMES(readdatapath)                                  ;
hdfdir      =["vid_true",'vid_pred','bnd']                               ; 
permorder   =[4,2,3,1]                                                   ; 
values      =[]                                                          ;
for i=1:size(filenames,2)
    vid_true    =permute(h5read(fullfile(readdatapath,filenames(i)),strcat('/',hdfdir(1))),permorder);
    vid_pred    =permute(h5read(fullfile(readdatapath,filenames(i)),strcat('/',hdfdir(2))),permorder);
    values      =[values;getMAXNORM(vid_true-vid_pred)]                                              ;
end
writematrix(values,savefilename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FUNCTION DEFINITIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function RDATAPATH=getDATAPATH(NET,TYPE)
        netpath          =fullfile('..','..','networks',NET             );
        if strcmp(TYPE,'TESTVID')
            RDATAPATH    =fullfile(netpath,'videos'  ,'test'      ,'hdf');
        else
            if strcmp(TYPE,'CONTVID')
                RDATAPATH=fullfile(netpath,'videos'  ,'prediction','hdf');
            end
        end
    end

    function FILENAMES=getFILENAMES(READDATAPATH)
        dirlist  =dir(READDATAPATH)                                                                     ;
        FILENAMES=string({dirlist(4:end).name})                                                         ;
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

end

