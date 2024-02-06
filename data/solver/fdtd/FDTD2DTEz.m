%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO                      %%%
%%% COMPUTATIONAL EM LAB                          %%%
%%% DEEP LEARNING PROJECT                         %%%
%%% TEz PROPAGATION/SCATTERING TF/SF PML BOUNDARY %%%
%%% by: OAMEED NOAKOASTEEN                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function time=FDTD2DTEz(TYPE,SPEC,savefilename)
addpath(fullfile('..','..','lib','fdtd2dtez'))

eps0        =8.85419*10^-12                                                    ; % FREE SPACE   : PERMITTIVITY
mu0         =4*pi*10^-7                                                        ; % FREE SPACE   : PERMEABILITY
nc          =40                                                                ; % MESH         : NUMBER OF CELLS PER WAVELENGTH
CS          =1/sqrt(2)                                                         ; % MESH         : COURANT STABILITY FACTOR
A           =1                                                                 ; % PULSE        : AMPLITUDE OF THE PULSE
fmax        =2*10^9                                                            ; % PULSE        : MAXIMUM FREQUENCY
Te          =6*10^-9                                                           ; % PULSE        : END OF THE TIME DOMAIN PULSE
pulse       =@(A,T,T0,TAU) A.*exp(-((T-T0)./TAU).^2)                           ; % PULSE        : PULSE SHAPE
L           =1.5                                                               ; % DOMAIN DESIGN: GEOMETRIC LENGTH OF DOMAIN (m)
PMLF        =0.25                                                              ; % DOMAIN DESIGN: ALLOCATION OF PML
TFSFB       =0.35                                                              ; % DOMAIN DESIGN: LOCATION OF BEGINNING OF TFSF BOUNDARY
TFSFE       =0.65                                                              ; % DOMAIN DESIGN: LOCATION OF BEGINNING OF TFSF BOUNDARY
ViB         =0.30                                                              ; % DOMAIN DESIGN: BEGINNING OF VIEW
ViE         =0.70                                                              ; % DOMAIN DESIGN: END OF VIEW
Npml        =2                                                                 ; % PML          : ORDER OF THE POWER-INCREASING SIGMA FUNCTION
R0          =10^-8                                                             ; % PML          : REQUIRED LEVEL OF REFLECTION FROM PML
NPMLCaux    =50                                                                ; % AUX 1D WAVE  : NUMBER OF PML CELLS FOR AUX WAVE
SIGMA       =10^8                                                              ; % MAT PROP     : CONDUCTIVITY
Nt          =400                                                               ; % FDTD LOOP    : NUMBER OF TIME-STEPS
ZMAX        =10                                                                ; % GRAPHICS     :
CAXIS       =2                                                                 ; % GRAPHICS     :
FRAC        =0.5                                                               ; % GRAPHICS     : FRACTION OF T0 TO DISREGARD
COUNTER     =floor(Nt/10:Nt/10:Nt)                                             ;

%%% PRE-PROCESSINGS %%%
c0      =1/sqrt(mu0*eps0)                                                      ; % FREE SPACE   : SPEED OF LIGHT (m/s)
LAMmin  =c0/fmax                                                               ; % MESH         : SMALLEST LAMBDA
delta   =LAMmin/nc                                                             ; % MESH         : DX=DY=DZ=DELTA
dt      =delta*CS/c0                                                           ; % MESH/PULSE   : COURANT STABILITY CONDITION
THETA   =SPEC(1)                                                               ; % PULSE        : ANGLE OF PROPAGATION (Rad)
TFSFCb  =floor((TFSFB*L)/delta)                                                ; % DOMAIN DESIGN: BEGINNING CELL INDEX OF TF REGION
TFSFCe  =floor((TFSFE*L)/delta)                                                ; % DOMAIN DESIGN: END CELL INDEX OF TF REGION

PSFx1   =floor((SPEC(2)*L)/delta)                                              ; % DOMAIN DESIGN: X LOCATION OF FIRST  POINT SOURCE
PSFy1   =floor((SPEC(3)*L)/delta)                                              ; % DOMAIN DESIGN: Y LOCATION OF FIRST  POINT SOURCE
PSFx2   =floor((SPEC(4)*L)/delta)                                              ; % DOMAIN DESIGN: X LOCATION OF SECOND POINT SOURCE
PSFy2   =floor((SPEC(5)*L)/delta)                                              ; % DOMAIN DESIGN: Y LOCATION OF SECOND POINT SOURCE
NPMLC   =floor((PMLF *L)/delta)                                                ; % PML          : NUMBER OF PML CELLS
SIGMAMAX=-((Npml+1)*eps0*c0*log(R0))/(2*delta*NPMLC)                           ; % PML          : POWER-INCREASING FUNCTION COEFFICIENT
auxB    =TFSFCb                                                                ; % AUX 1D WAVE  : BEGINNING CELL NUMBER OF TF/SF
auxE    =TFSFCe                                                                ; % AUX 1D WAVE  : ENDING CELL NUMBER OF TF/SF
NumAUXc =floor(sqrt(2)*(auxE-auxB))+10+NPMLCaux                                ; % AUX 1D WAVE  : LENGTH OF AUXILIARY DOMAIN
XlowL   =floor((ViB*L)/delta)                                                  ; % GRAPHICS
XupL    =floor((ViE*L)/delta)                                                  ; % GRAPHICS
YlowL   =floor((ViB*L)/delta)                                                  ; % GRAPHICS
YupL    =floor((ViE*L)/delta)                                                  ; % GRAPHICS

%%% INITIALIZE PULSE %%%
tau =sqrt(-log(0.1))/(pi*fmax)                                                 ;
t0  =sqrt(20)*tau                                                              ;
t   =0:dt:Te                                                                   ;
P   =pulse(A,t,t0,tau)                                                         ;
STEP=((FRAC)*t0)/dt                                                            ;

%%% INITIALZE FIELD ARRAYS %%%

% Hz(:,:,1)   TOTAL Hz
% Hz(:,:,2)   PML Hzx
% Hz(:,:,3)   PML Hzy

l   =0:delta:L                                                                 ;
Ndx =size(l,2)-1                                                               ;
Ndy =size(l,2)-1                                                               ;
Ex  =zeros(Ndy+1,Ndx         )                                                 ;
Ey  =zeros(Ndy  ,Ndx+1       )                                                 ;
Hz  =zeros(Ndy  ,Ndx      ,3 )                                                 ;
EX  =zeros(Ndy+1,Ndx      ,Nt)                                                 ; % DATA OUTPUT: CONTAINS DATA FOR THE ENTIRE SIMULATION
EY  =zeros(Ndy  ,Ndx+1    ,Nt)                                                 ; % DATA OUTPUT: CONTAINS DATA FOR THE ENTIRE SIMULATION
HZ  =zeros(Ndy  ,Ndx      ,Nt)                                                 ; % DATA OUTPUT: CONTAINS DATA FOR THE ENTIRE SIMULATION
if strcmp(TYPE,'type1')
    Einc=zeros(1,NumAUXc+1   )                                                 ; % INCIDENT ELECTRIC FIELD FOR AUX 1D WAVE
    Hinc=zeros(1,NumAUXc     )                                                 ; % INCIDENT MAGNETIC FIELD FOR AUX 1D WAVE
end

returns=get_fdtd_coefficients(false                    ,...
                              {[size(Ex,1),size(Ex,2)] ,...
                               [size(Ey,1),size(Ey,2)] ,...
                               [size(Hz,1),size(Hz,2)]},...
                               SPEC                    ,...
                               [eps0,mu0,SIGMA,L,delta ,...
                                LAMmin,NPMLC,SIGMAMAX  ,...
                                Npml,NPMLC,NumAUXc     ,...
                                NPMLCaux,dt                ])                  ;
Cex    =returns{1}                                                             ;
Cey    =returns{2}                                                             ;
Chz    =returns{3}                                                             ;
MPex   =returns{4}                                                             ;

if strcmp(TYPE,'type1')
    returns=get_tfsf_regions(SPEC,[TFSFCe,TFSFCb])                             ;
    O1     =returns{1}                                                         ;
    O2     =returns{2}                                                         ;
    O3     =returns{3}                                                         ;
    O4     =returns{4}                                                         ;
    I1     =returns{5}                                                         ;
    I4     =returns{6}                                                         ;

    returns=get_fdtd_coefficients(true                     ,...
                                  {[size(Ex,1),size(Ex,2)] ,...
                                   [size(Ey,1),size(Ey,2)] ,...
                                   [size(Hz,1),size(Hz,2)]},...
                                   SPEC                    ,...
                                   [eps0,mu0,SIGMA,L,delta ,...
                                    LAMmin,NPMLC,SIGMAMAX  ,...
                                    Npml,NPMLC,NumAUXc     ,...
                                    NPMLCaux,dt                ])              ;
    Ceaux=returns{1}                                                           ;
    Chaux=returns{2}                                                           ;
end



%%% FDTD UPDATE LOOP %%%
if strcmp(TYPE,'type1')
    time=[]                                                                    ;
    for T=1:Nt
        tic
        %%% AUX 1D WAVE UPDATE
        for i=1:size(Hinc,2)
            Hinc(1,i)=Chaux(2,i)*Hinc(1,i)+Chaux(1,i)*(Einc(1,i+1)-Einc(1,i))  ;
        end
        for i=2:size(Einc,2)-1
            Einc(1,i)=Ceaux(1,i)*Einc(1,i)+Ceaux(2,i)*(Hinc(1,i)-Hinc(1,i-1))  ;
        end
        if T<=size(P,2)
            Einc(1,1)=Ceaux(3,1)*P(T)                                          ;
        end
        %%% UPDATE CONNECTIVITY VALUES
        for i=1:(TFSFCe-TFSFCb)+1
            O1(i,5)=(1-O1(i,7))*Einc(O1(i,3))+O1(i,7)*Einc(O1(i,3)+1)          ;
            O2(i,5)=(1-O2(i,7))*Einc(O2(i,3))+O2(i,7)*Einc(O2(i,3)+1)          ;
            O2(i,6)=(1-O2(i,8))*Hinc(O2(i,4))+O2(i,8)*Hinc(O2(i,4)+1)          ;
            O3(i,5)=(1-O3(i,7))*Einc(O3(i,3))+O3(i,7)*Einc(O3(i,3)+1)          ;
            O3(i,6)=(1-O3(i,8))*Hinc(O3(i,4))+O3(i,8)*Hinc(O3(i,4)+1)          ;
            O4(i,5)=(1-O4(i,7))*Einc(O4(i,3))+O4(i,7)*Einc(O4(i,3)+1)          ;
            I1(i,6)=(1-I1(i,8))*Hinc(I1(i,4))+I1(i,8)*Hinc(I1(i,4)+1)          ;
            I4(i,6)=(1-I4(i,8))*Hinc(I4(i,4))+I4(i,8)*Hinc(I4(i,4)+1)          ;
        end
        for j=1:Ndy
            for i=1:Ndx
                if (NPMLC<i && i<size(Hz,2)-NPMLC)&&(NPMLC<j && j<size(Hz,1)-NPMLC)
                    Hz(j,i,1)=Chz(j,i,1)* Hz(j,i,1)         +...
                              Chz(j,i,2)*(Ex(j+1,i)-Ex(j,i))+...
                              Chz(j,i,3)*(Ey(j,i+1)-Ey(j,i))                   ;
                    if ismember([i,j],O1(:,[1,2]),'rows')
                        [~,ind]  =ismember([i,j],O1(:,[1,2]),'rows')           ;
                        Hz(j,i,1)=Hz(j,i,1)-...
                            (dt/(mu0*delta))*O1(ind,5)*(-sin(THETA))           ;
                    end
                    if ismember([i,j],O2(:,[1,2]),'rows')
                        [~,ind]  =ismember([i,j],O2(:,[1,2]),'rows')           ;
                        Hz(j,i,1)=Hz(j,i,1)-...
                            (dt/(mu0*delta))*O2(ind,5)*cos(THETA)              ;
                    end
                    if ismember([i,j],O3(:,[1,2]),'rows')
                        [~,ind]  =ismember([i,j],O3(:,[1,2]),'rows')           ;
                        Hz(j,i,1)=Hz(j,i,1)+...
                            (dt/(mu0*delta))*O3(ind,5)*(-sin(THETA))           ;
                    end
                    if ismember([i,j],O4(:,[1,2]),'rows')
                        [~,ind]  =ismember([i,j],O4(:,[1,2]),'rows')           ;
                        Hz(j,i,1)=Hz(j,i,1)+...
                            (dt/(mu0*delta))*O4(ind,5)*cos(THETA)              ;
                    end
                else
                    Hz(j,i,2)=Chz(j,i,5)* Hz(j,i,2)         +...
                              Chz(j,i,8)*(Ey(j,i+1)-Ey(j,i))                   ;
                    Hz(j,i,3)=Chz(j,i,6)* Hz(j,i,3)         +...
                              Chz(j,i,7)*(Ex(j+1,i)-Ex(j,i))                   ;
                    Hz(j,i,1)=Hz(j,i,2)+Hz(j,i,3)                              ;
                end
            end
        end
        for j=2:Ndy
            for i=1:Ndx
                Ex(j,i)=Cex(j,i,1)* Ex(j,i)           +...
                        Cex(j,i,2)*(Hz(j,i)-Hz(j-1,i))                         ;
                if ismember([i,j],O3(:,[1,2]),'rows')
                    [~,ind]=ismember([i,j],O3(:,[1,2]),'rows')                 ;
                    Ex(j,i)=Ex(j,i)+...
                        (dt/(eps0*delta))*O3(ind,6)                            ;
                end
                if ismember([i,j],I1(:,[1,2]),'rows')
                    [~,ind]=ismember([i,j],I1(:,[1,2]),'rows')                 ;
                    Ex(j,i)=Ex(j,i)-...
                        (dt/(eps0*delta))*I1(ind,6)                            ;
                end
            end
        end
        for j=1:Ndy
            for i=2:Ndx
                Ey(j,i)=Cey(j,i,1)* Ey(j,i)           +...
                        Cey(j,i,2)*(Hz(j,i)-Hz(j,i-1))                         ;
                if ismember([i,j],O2(:,[1,2]),'rows')
                    [~,ind]=ismember([i,j],O2(:,[1,2]),'rows')                 ;
                    Ey(j,i)=Ey(j,i)-...
                        (dt/(eps0*delta))*O2(ind,6)                            ;
                end
                if ismember([i,j],I4(:,[1,2]),'rows')
                    [~,ind]=ismember([i,j],I4(:,[1,2]),'rows')                 ;
                    Ey(j,i)=Ey(j,i)+...
                        (dt/(eps0*delta))*I4(ind,6)                            ;
                end
            end
        end
        for i=1:size(Hz,2)
            for j=1:size(Hz,1)
                EX(j,i,T)=Ex(j,i)                                              ;
                EY(j,i,T)=Ey(j,i)                                              ;
                HZ(j,i,T)=Hz(j,i,1)                                            ;
            end
        end
        if ismember(T,COUNTER)
            disp([' COMPLETED ',' ',num2str(floor(100*(T/Nt))),' ','%'])
        end
    TIME=toc                                                                   ;
    time=[time;TIME]                                                           ;
    end    
else
    if strcmp(TYPE,'type2') || strcmp(TYPE,'type3')
        time=[]                                                                ;
        for T=1:Nt
            tic
            for j=1:Ndy
                for i=1:Ndx
                    if (NPMLC<i && i<size(Hz,2)-NPMLC)&&(NPMLC<j && j<size(Hz,1)-NPMLC)
                        Hz(j,i,1)=Chz(j,i,1)* Hz(j,i,1)         +...
                                  Chz(j,i,2)*(Ex(j+1,i)-Ex(j,i))+...
                                  Chz(j,i,3)*(Ey(j,i+1)-Ey(j,i))               ;
                    else
                        Hz(j,i,2)=Chz(j,i,5)* Hz(j,i,2)         +...
                                  Chz(j,i,8)*(Ey(j,i+1)-Ey(j,i))               ;
                        Hz(j,i,3)=Chz(j,i,6)* Hz(j,i,3)         +...
                                  Chz(j,i,7)*(Ex(j+1,i)-Ex(j,i))               ;
                        Hz(j,i,1)=Hz(j,i,2)+Hz(j,i,3)                          ;
                    end
                end
            end
            %%% APPLY SOURCE
            if strcmp(TYPE,'type2')
                if T<=size(P,2)
                    Hz(PSFx1,PSFy1)=Hz(PSFx1,PSFy1)+Chz(PSFx1,PSFy1,4)*P(T)    ;
                end
            else
                if strcmp(TYPE,'type3')
                    if T<=size(P,2)
                        Hz(PSFx1,PSFy1)=Hz(PSFx1,PSFy1)+Chz(PSFx1,PSFy1,4)*P(T);
                        Hz(PSFx2,PSFy2)=Hz(PSFx2,PSFy2)+Chz(PSFx2,PSFy2,4)*P(T);
                    end
                end
            end
            for j=2:Ndy
                for i=1:Ndx
                    Ex(j,i)=Cex(j,i,1)* Ex(j,i)           +...
                            Cex(j,i,2)*(Hz(j,i)-Hz(j-1,i))                     ;
                end
            end
            for j=1:Ndy
                for i=2:Ndx
                    Ey(j,i)=Cey(j,i,1)* Ey(j,i)           +...
                            Cey(j,i,2)*(Hz(j,i)-Hz(j,i-1))                     ;
                end
            end
            for i=1:size(Hz,2)
                for j=1:size(Hz,1)
                    EX(j,i,T)=Ex(j,i)                                          ;
                    EY(j,i,T)=Ey(j,i)                                          ;
                    HZ(j,i,T)=Hz(j,i,1)                                        ;
                end
            end
            if ismember(T,COUNTER)
                disp([' COMPLETED ',' ',num2str(floor(100*(T/Nt))),' ','%'])
            end
        TIME=toc                                                               ;
        time=[time;TIME]                                                       ;
        end
    end
end

time=mean(time)                                                                ;

wHDF(savefilename                 ,...
    {'Ex_FIELD'                   ,...
     'Ey_FIELD'                   ,...
     'Hz_FIELD'                   ,...
     'boundary'}                  ,...
    {EX(  XlowL:XupL,YlowL:YupL,:),...
     EY(  XlowL:XupL,YlowL:YupL,:),...
     HZ(  XlowL:XupL,YlowL:YupL,:),...
     MPex(XlowL:XupL,YlowL:YupL,2)    })

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FUNCTION DEFINITIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function wHDF(FILENAME,DIR,DATA)
        for index=1:size(DIR,2)
            h5create(FILENAME,strcat('/',DIR{index}),size(DATA{index}))      ;
            h5write (FILENAME,strcat('/',DIR{index}),DATA{index}      )      ;
        end
    end

end

