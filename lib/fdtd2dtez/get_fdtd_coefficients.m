%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO                      %%%
%%% COMPUTATIONAL EM LAB                          %%%
%%% EMULATING FDTD USING TRANSFORMER & GNN        %%%
%%% TEz PROPAGATION/SCATTERING TF/SF PML BOUNDARY %%%
%%% GENERATE FDTD COEFFICIENTS                    %%%
%%% by: OAMEED NOAKOASTEEN                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cex(:,:,1)  Cexe
% Cex(:,:,2)  Cexh
% Cex(:,:,3)  Cexj
% Cey(:,:,1)  Ceye
% Cey(:,:,2)  Ceyh
% Cey(:,:,3)  Ceyj
% Chz(:,:,1)  Chzhz  (FREE SPACE)
% Chz(:,:,2)  Chzex  (FREE SPACE)
% Chz(:,:,3)  Chzey  (FREE SPACE)
% Chz(:,:,4)  Chzm   (FREE SPACE)
% Chz(:,:,5)  Chzhzx (PML)
% Chz(:,:,6)  Chzhzy (PML)
% Chz(:,:,7)  Chzex  (PML)
% Chz(:,:,8)  Chzey  (PML)

function returns=get_fdtd_coefficients(AUX,SIZE,SPEC,PARAMS)

MPreturns=get_materials(AUX,SIZE,SPEC,PARAMS)           ;
delta    =PARAMS(5)                                     ;
NumAUXc  =PARAMS(11)                                    ;
dt       =PARAMS(13)                                    ;

if ~AUX
    MPex=MPreturns{1}                                   ;
    MPey=MPreturns{2}                                   ;
    MPhz=MPreturns{3}                                   ;
    Cex =zeros(SIZE{1}(1),SIZE{1}(2),3)                 ;
    Cey =zeros(SIZE{2}(1),SIZE{2}(2),3)                 ;
    Chz =zeros(SIZE{3}(1),SIZE{3}(2),8)                 ;
    % FIELD COEFFICIENT VALUES 
    Cex(:,:,1)=(2.*MPex(:,:,1)-dt.*MPex(:,:,2))./...
        (2.*MPex(:,:,1)+dt.*MPex(:,:,2))                ;
    Cex(:,:,2)=(2*dt)./...
        ((2.*MPex(:,:,1)+dt.*MPex(:,:,2)).*delta)       ;
    Cex(:,:,3)=-(2*dt)./...
        (2.*MPex(:,:,1)+dt.*MPex(:,:,2))                ;
    Cey(:,:,1)=(2.*MPey(:,:,1)-dt.*MPey(:,:,2))./...
        (2.*MPey(:,:,1)+dt.*MPey(:,:,2))                ;
    Cey(:,:,2)=-(2*dt)./...
        ((2.*MPey(:,:,1)+dt.*MPey(:,:,2)).*delta)       ;
    Cey(:,:,3)=-(2*dt)./...
        (2.*MPey(:,:,1)+dt.*MPey(:,:,2))                ;
    Chz(:,:,1)=((2.*MPhz(:,:,1)-dt.*MPhz(:,:,2))./...
        (2.*MPhz(:,:,1)+dt.*MPhz(:,:,2)))               ;
    Chz(:,:,2)=(2*dt)./...
        ((2.*MPhz(:,:,1)+dt.*MPhz(:,:,2)).*delta)       ;
    Chz(:,:,3)=-(2*dt)./...
        ((2.*MPhz(:,:,1)+dt.*MPhz(:,:,2)).*delta)       ;
    Chz(:,:,4)=-(2*dt)./...
        (2.*MPhz(:,:,1)+dt.*MPhz(:,:,2))                ;
    Chz(:,:,5)=(2.*MPhz(:,:,1)-dt.*MPhz(:,:,3))./...
        (2.*MPhz(:,:,1)+dt.*MPhz(:,:,3))                ;
    Chz(:,:,6)=(2.*MPhz(:,:,1)-dt.*MPhz(:,:,4))./...
        (2.*MPhz(:,:,1)+dt.*MPhz(:,:,4))                ;
    Chz(:,:,7)=(2*dt)./...
        ((2.*MPhz(:,:,1)+dt.*MPhz(:,:,4)).*delta)       ;
    Chz(:,:,8)=-(2*dt)./...
        ((2.*MPhz(:,:,1)+dt.*MPhz(:,:,3)).*delta)       ;
    
    returns={Cex,Cey,Chz,MPex}                          ;
    
else
    MPeaux=MPreturns{1}                                 ;    
    MPhaux=MPreturns{2}                                 ;
    Ceaux =zeros(3,NumAUXc+1)                           ; % FOR AUX 1D WAVE
    Chaux =zeros(3,NumAUXc  )                           ; % FOR AUX 1D WAVE
    % FIELD COEFFICIENT VALUES FOR AUX 1D WAVE
    Ceaux(1,:)=(2.*MPeaux(1,:)-dt.*MPeaux(2,:))./...
        (2.*MPeaux(1,:)+dt.*MPeaux(2,:))                ;
    Ceaux(2,:)=-(2*dt)./...
        ((2.*MPeaux(1,:)+dt.*MPeaux(2,:)).*delta)       ;
    Ceaux(3,:)=-(2*dt)./...
        (2.*MPeaux(1,:)+dt.*MPeaux(2,:))                ;
    Chaux(1,:)=-(2*dt)./...
        ((2.*MPhaux(1,:)+dt.*MPhaux(2,:)).*delta)       ;
    Chaux(2,:)=(2.*MPhaux(1,:)-dt.*MPhaux(2,:))./...
        (2.*MPhaux(1,:)+dt.*MPhaux(2,:))                ;
    Chaux(3,:)=-(2*dt)./...
        (2.*MPhaux(1,:)+dt.*MPhaux(2,:))                ;
    
    returns={Ceaux,Chaux}                               ;
    
end

end

