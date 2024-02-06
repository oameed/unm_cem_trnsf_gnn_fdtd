%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO                      %%%
%%% COMPUTATIONAL EM LAB                          %%%
%%% EMULATING FDTD USING TRANSFORMER & GNN        %%%
%%% TEz PROPAGATION/SCATTERING TF/SF PML BOUNDARY %%%
%%% GENERATE MATERIAL DATA STRUCTURES             %%%
%%% by: OAMEED NOAKOASTEEN                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MPex(:,:,1) PERMATIVITTY (eps_x)
% MPex(:,:,2) ELECTRIC CONDUCTIVITY (sigma_e_x)/ PML (sigma_p_e_y)
% MPey(:,:,1) PERMATIVITTY (eps_y)
% MPey(:,:,2) ELECTRIC CONDUCTIVITY (sigma_e_y)/ PML (sigma_p_e_x)
% MPhz(:,:,1) PERMEABILITY (mu_z) 
% MPhz(:,:,2) FREE SPACE MAGNETIC CONDUCTIVITY (sigma_m_z)
% MPhz(:,:,3) PML MAGNETIC CONDUCTIVITY (sigma_p_m_x)
% MPhz(:,:,4) PML MAGNETIC CONDUCTIVITY (sigma_p_m_y)

% FOR AUX 1D WAVE TEz CASE (Ey,Hz):

% MPe(1,:) PERMATIVITTY (eps_y) 
% MPe(2,:) ELECTRIC CONDUCTIVITY (sigma_e_y)/ PML (sigma_p_e_x)
% MPh(1,:) PERMEABILITY (mu_z) 
% MPh(2,:) PML MAGNETIC CONDUCTIVITY (sigma_p_m_x)

function returns=get_materials(AUX,SIZE,SPEC,PARAMS)

eps0    =PARAMS(1)                                                   ;
mu0     =PARAMS(2)                                                   ;
SIGMA   =PARAMS(3)                                                   ;
L       =PARAMS(4)                                                   ;
delta   =PARAMS(5)                                                   ;
LAMmin  =PARAMS(6)                                                   ;
NPMLC   =PARAMS(7)                                                   ;
SIGMAMAX=PARAMS(8)                                                   ;
Npml    =PARAMS(9)                                                   ;
NPMLC   =PARAMS(10)                                                  ;
NumAUXc =PARAMS(11)                                                  ;
NPMLCaux=PARAMS(12)                                                  ;
circcoef=0.60                                                        ;

if ~AUX
    MPex  =zeros(SIZE{1}(1),SIZE{1}(2),2)                            ;
    MPey  =zeros(SIZE{2}(1),SIZE{2}(2),2)                            ;
    MPhz  =zeros(SIZE{3}(1),SIZE{3}(2),4)                            ;
    % MATERIAL PROPERTIES OF FREE SPACE / NON-PML REGION
    MPex(:,:,1)=1*eps0                                               ;
    MPey(:,:,1)=1*eps0                                               ;
    MPhz(:,:,1)=1*mu0                                                ;    
    %%% GENERATE OBJECTS %%%
    if SPEC(11)==1        
        objclx=floor(SPEC(7)*L/delta);
        objcly=floor(SPEC(8)*L/delta);
        side=floor(((1/sqrt(2))*SPEC(12)*LAMmin)/delta)              ;
        if mode(side,2)==0
            OBJBx=objclx-side/2+1                                    ;
            OBJEx=objclx+side/2-1                                    ;
            OBJBy=objcly-side/2+1                                    ;
            OBJEy=objcly+side/2-1                                    ;
        else
            OBJBx=objclx-floor(side/2)+1                             ;
            OBJEx=objclx+floor(side/2)-1                             ;
            OBJBy=objcly-floor(side/2)+1                             ;
            OBJEy=objcly+floor(side/2)-1                             ;
        end
        MPex(OBJBx:OBJEx,OBJBy:OBJEy,2)=SIGMA                        ;
        MPey(OBJBx:OBJEx,OBJBy:OBJEy,2)=SIGMA                        ;
    else
        if SPEC(11)==2
            objclx=floor(SPEC(7)*L/delta)                            ;
            objcly=floor(SPEC(8)*L/delta)                            ;
            for i=1:size(MPex,1)
                for j=1:size(MPex,2)
                    if (i-objclx)^2+(j-objcly)^2<=(((SPEC(12)*LAMmin)/delta)*circcoef)^2
                        MPex(i,j,2)=SIGMA                            ;
                        MPey(i,j,2)=SIGMA                            ;
                    end
                end
            end
        else
            if SPEC(11)==3
                objclx=floor(SPEC(7)*L/delta)                        ;
                objcly=floor(SPEC(8)*L/delta)                        ;
                side=floor(((1/sqrt(2))*SPEC(12)*LAMmin)/delta)      ;
                if mode(side,2)==0
                    OBJBx=objclx-side/2+1                            ;
                    OBJEx=objclx+side/2-1                            ;
                    OBJBy=objcly-side/2+1                            ;
                    OBJEy=objcly+side/2-1                            ;
                else
                    OBJBx=objclx-floor(side/2)+1                     ;
                    OBJEx=objclx+floor(side/2)-1                     ;
                    OBJBy=objcly-floor(side/2)+1                     ;
                    OBJEy=objcly+floor(side/2)-1                     ;
                end
                MPex(OBJBx:OBJEx,OBJBy:OBJEy,2)=SIGMA                ;
                MPey(OBJBx:OBJEx,OBJBy:OBJEy,2)=SIGMA                ;
                objclx=floor(SPEC(9)*L/delta)                        ;
                objcly=floor(SPEC(10)*L/delta)                       ;
                for i=1:size(MPex,1)
                    for j=1:size(MPex,2)
                        if (i-objclx)^2+(j-objcly)^2<=(((SPEC(13)*LAMmin)/delta)*circcoef)^2
                            MPex(i,j,2)=SIGMA                        ;
                            MPey(i,j,2)=SIGMA                        ;
                        end
                    end
                end
            end
        end
    end        
        
    % MATERIAL PROPERTIES OF PML REGION
    ib=2                                                             ;
    ie=NPMLC+1                                                       ;
    for i=ib:ie
        MPey(:,i,2)  =SIGMAMAX.*((1/(ie-ib))*(-i+ie))^Npml           ;
        MPhz(:,i-1,3)=(mu0/eps0)*MPey(1,i,2)                         ;
        MPex(i,:,2)  =SIGMAMAX.*((1/(ie-ib))*(-i+ie))^Npml           ;
        MPhz(i-1,:,4)=(mu0/eps0)*MPex(i,1,2)                         ;
    end
    ib=size(MPex,1)-NPMLC                                            ;
    ie=size(MPex,1)-1                                                ;
    for i=ib:ie    
        MPex(i,:,2)=SIGMAMAX.*((1/(ie-ib))*(i-ib))^Npml              ;
        MPhz(i,:,4)=(mu0/eps0)*MPex(i,1,2)                           ;
    end
    ib=size(MPey,2)-NPMLC                                            ;
    ie=size(MPey,2)-1                                                ;
    for i=ib:ie 
        MPey(:,i,2)=SIGMAMAX.*((1/(ie-ib))*(i-ib))^Npml              ;
        MPhz(:,i,3)=(mu0/eps0)*MPey(1,i,2)                           ;
    end
    
    returns={MPex,MPey,MPhz}                                         ;
    
else
    MPeaux     =zeros(2,NumAUXc+1)                                   ; % FOR AUX 1D WAVE
    MPhaux     =zeros(2,NumAUXc  )                                   ; % FOR AUX 1D WAVE
    MPeaux(1,:)=1*eps0                                               ; % FOR AUX 1D WAVE
    MPhaux(1,:)=1*mu0                                                ; % FOR AUX 1D WAVE
    % MATERIAL PROPERTIES OF PML REGION FOR AUX 1D WAVE
    ib         =size(MPeaux,2)-NPMLCaux                              ;
    ie         =size(MPeaux,2)-1                                     ;
    for i=ib:ie
        MPeaux(2,i)=SIGMAMAX.*((1/(ie-ib))*(i-ib))^Npml              ;
        MPhaux(2,i)=(mu0/eps0)*MPeaux(2,i)                           ;
    end
    
    returns={MPeaux,MPhaux}                                          ;
    
end

end

