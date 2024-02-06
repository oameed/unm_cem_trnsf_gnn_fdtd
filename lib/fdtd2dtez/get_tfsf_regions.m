%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UNIVERSITY OF NEW MEXICO                      %%%
%%% COMPUTATIONAL EM LAB                          %%%
%%% EMULATING FDTD USING TRANSFORMER & GNN        %%%
%%% TEz PROPAGATION/SCATTERING TF/SF PML BOUNDARY %%%
%%% GENERATE TFSF O1-O4/I1-I4 REGIONS             %%%
%%% by: OAMEED NOAKOASTEEN                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% d: DOT PRODUCT OF UNIT VECTOR OF DIRECTION AND POSITION OF E/H VECTORS
% <REGION>(1,:) <REGION> X-AXIS INDEX
% <REGION>(2,:) <REGION> Y-AXIS INDEX
% <REGION>(3,:) <REGION> E-FIELD INDEX ON 1D AUX WAVE 
% <REGION>(4,:) <REGION> H-FIELD INDEX ON 1D AUX WAVE 
% <REGION>(5,:) <REGION> E-FIELD UPDATED INCIDENCE VALUE FROM 1D AUX WAVE
% <REGION>(6,:) <REGION> H-FIELD UPDATED INCIDENCE VALUE FROM 1D AUX WAVE
% <REGION>(7,:) <REGION> E-FIELD FRACTIONAL REMAINDER OF 'd' FOR INTERPOLATION
% <REGION>(8,:) <REGION> H-FIELD FRACTIONAL REMAINDER OF 'd' FOR INTERPOLATION

function returns=get_tfsf_regions(SPEC,PARAMS)

THETA =SPEC(1)                                        ;
TFSFCe=PARAMS(1)                                      ;
TFSFCb=PARAMS(2)                                      ;

O1=zeros((TFSFCe-TFSFCb)+1,8)                         ;
O2=zeros((TFSFCe-TFSFCb)+1,8)                         ;
O3=zeros((TFSFCe-TFSFCb)+1,8)                         ;
O4=zeros((TFSFCe-TFSFCb)+1,8)                         ;
I1=zeros((TFSFCe-TFSFCb)+1,8)                         ;
I4=zeros((TFSFCe-TFSFCb)+1,8)                         ;
for i=1:(TFSFCe-TFSFCb)+1
    O1(i,1)=(i-1)+TFSFCb                              ;
    O1(i,2)=TFSFCb-1                                  ;
    O2(i,1)=TFSFCe+1                                  ;
    O2(i,2)=(i-1)+TFSFCb                              ;
    O3(i,1)=(i-1)+TFSFCb                              ;
    O3(i,2)=TFSFCe+1                                  ;
    O4(i,1)=TFSFCb-1                                  ;
    O4(i,2)=(i-1)+TFSFCb                              ;
    I1(i,1)=(i-1)+TFSFCb                              ;
    I1(i,2)=TFSFCb                                    ;
    I4(i,1)=TFSFCb                                    ;
    I4(i,2)=(i-1)+TFSFCb                              ;
end
for i=1:(TFSFCe-TFSFCb)+1
    % O1 REGION
    d      =((O1(i,1)-(TFSFCb-1))-1/2)*cos(THETA)+...
            ((O1(i,2)-(TFSFCb-1))-0  )*sin(THETA)     ;
    O1(i,7)=d-floor(d)                                ;
    O1(i,3)=floor(d)+5                                ;
    % O2 REGION
    d      =((O2(i,1)-(TFSFCb-1))-1  )*cos(THETA)+...
            ((O2(i,2)-(TFSFCb-1))-1/2)*sin(THETA)     ;
    O2(i,7)=d-floor(d)                                ;
    O2(i,3)=floor(d)+5                                ;
    d      =((O2(i,1)- TFSFCb   )-1/2)*cos(THETA)+...
            ((O2(i,2)-(TFSFCb-1))-1/2)*sin(THETA)+1/2 ;
    O2(i,8)=d-floor(d)                                ;
    O2(i,4)=floor(d)+5                                ;
    % O3 REGION
    d      =((O3(i,1)-(TFSFCb-1))-1/2)*cos(THETA)+...
            ((O3(i,2)-(TFSFCb-1))-1  )*sin(THETA)     ;
    O3(i,7)=d-floor(d)                                ;
    O3(i,3)=floor(d)+5                                ;
    d      =((O3(i,1)-(TFSFCb-1))-1/2)*cos(THETA)+...
            ((O3(i,2)-(TFSFCb-1))-1/2)*sin(THETA)+1/2 ;
    O3(i,8)=d-floor(d)                                ;
    O3(i,4)=floor(d)+5                                ;
    % O4 REGION
    d      =((O4(i,1)-(TFSFCb-1))-0  )*cos(THETA)+...
            ((O4(i,2)-(TFSFCb-1))-1/2)*sin(THETA)     ;
    O4(i,7)=d-floor(d)                                ;
    O4(i,3)=floor(d)+5                                ;
    % I1 REGION
    d      =((I1(i,1)-(TFSFCb-1))-1/2)*cos(THETA)+...
            ((I1(i,2)-(TFSFCb-1))-3/2)*sin(THETA)+1/2 ;
    I1(i,8)=d-floor(d)                                ;
    I1(i,4)=floor(d)+5                                ;
    % I4 REGION
    d      =( I4(i,1)-(TFSFCb-1) -3/2)*cos(THETA)+...
            ((I4(i,2)-(TFSFCb-1))-1/2)*sin(THETA)+1/2 ;
    I4(i,8)=d-floor(d)                                ;
    I4(i,4)=floor(d)+5                                ;
end

returns={O1,O2,O3,O4,I1,I4}                           ;

end

