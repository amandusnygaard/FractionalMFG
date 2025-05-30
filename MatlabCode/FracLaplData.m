p_cube =[
[0.1127, 0.1127, 0.1127];
[0.1127, 0.1127, 0.5000];
[0.1127, 0.1127, 0.8873];
[0.1127, 0.5000, 0.1127];
[0.1127, 0.5000, 0.5000];
[0.1127, 0.5000, 0.8873];
[0.1127, 0.8873, 0.1127];
[0.1127, 0.8873, 0.5000];
[0.1127, 0.8873, 0.8873];
[0.5000, 0.1127, 0.1127];
[0.5000, 0.1127, 0.5000];
[0.5000, 0.1127, 0.8873];
[0.5000, 0.5000, 0.1127];
[0.5000, 0.5000, 0.5000];
[0.5000, 0.5000, 0.8873];
[0.5000, 0.8873, 0.1127];
[0.5000, 0.8873, 0.5000];
[0.5000, 0.8873, 0.8873];
[0.8873, 0.1127, 0.1127];
[0.8873, 0.1127, 0.5000];
[0.8873, 0.1127, 0.8873];
[0.8873, 0.5000, 0.1127];
[0.8873, 0.5000, 0.5000];
[0.8873, 0.5000, 0.8873];
[0.8873, 0.8873, 0.1127];
[0.8873, 0.8873, 0.5000];
[0.8873, 0.8873, 0.8873]];

p_T_6 = [
[0.5541, 0.4459];
[0.5541, 0.1081];
[0.8919, 0.4459];
[0.9084, 0.0916];
[0.9084, 0.8168];
[0.1832, 0.0916]];

p_T_12 =[
[0.7507, 0.2493];
[0.7507, 0.5014];
[0.4986, 0.2493];
[0.9369, 0.0631];
[0.9369, 0.8738];
[0.1262, 0.0631];
[0.6896, 0.6365];
[0.3635, 0.0531];
[0.9469, 0.3104];
[0.3635, 0.3104];
[0.6896, 0.0531];
[0.9469, 0.6365]];

p_I = [
0.5000, 
0.0820, 
0.9180, 
0.0159, 
0.9841, 
0.3379, 
0.6621, 
0.8067, 
0.1933];

w_I = [
0.1651,
0.0903,
0.0903,
0.0406,
0.0406,
0.1562,
0.1562,
0.1303,
0.1303];

% Create phi matrices
w_T_6 = zeros(6,1);
w_T_6(1) = 0.1117;
w_T_6(2) = w_T_6(1);
w_T_6(3) = w_T_6(1);
w_T_6(4) = 0.0550;
w_T_6(5) = w_T_6(4);
w_T_6(6) = w_T_6(4);
local = cell(1,6);
local{1} = @(x,y) 1-x;
local{2} = @(x,y) x-y;
local{3} = @(x,y) y;
local{4} = @(x,y) -(1-x);
local{5} = @(x,y) -(x-y);
local{6} = @(x,y) -y;
mat_loc = zeros(6);
for i = 1:6
    for j = 1:6
        mat_loc(i,j) = local{i}(p_T_6(j,1),p_T_6(j,2));
    end
end
W = w_T_6*(w_T_6');
M_aux = zeros(18);
N_aux = zeros(18);
L_aux = zeros(18);
phiB = zeros(9,36);
phiA = zeros(9,36);
phiD = zeros(9,36);
for i=1:3
    for j=1:3
        for k = 1:6
            for q=1:6
                M_aux( q + 6*(i-1) , k + 6*(j-1) ) =...
                W(q,k)*mat_loc(i,q)*mat_loc(j+3,k);
                N_aux( q + 6*(i-1) , k + 6*(j-1) ) =...
                W(q,k)*mat_loc(i,q)*mat_loc(j,q);
                L_aux( q + 6*(i-1) , k + 6*(j-1) ) =...
                W(q,k)*mat_loc(i+3,k)*mat_loc(j+3,k);
            end
        end
    end
end
for i=1:9
    [im, jm] = ind2sub([3 3] , i);
    im = 6*(im - 1) + 1;
    jm = 6*(jm - 1) + 1;
    phiB(i,:) = reshape( M_aux( im:im+5 , jm:jm+5 ) , 1 , [] );
    phiA(i,:) = reshape( N_aux( im:im+5 , jm:jm+5 ) , 1 , [] );
    phiD(i,:) = reshape( L_aux( im:im+5 , jm:jm+5 ) , 1 , [] );
end

w_cube = [
0.0214,
0.0343,
0.0214,
0.0343,
0.0549,
0.0343,
0.0214,
0.0343,
0.0214,
0.0343,
0.0549,
0.0343,
0.0549,
0.0878,
0.0549,
0.0343, 
0.0549,
0.0343,
0.0214,
0.0343,
0.0214,
0.0343,
0.0549,
0.0343,
0.0214,
0.0343,
0.0214 ];

% Create vpsi matrices
psi_D1 = cell(5,1);
psi_D1{1} = @(x,y,z) y-1;
psi_D1{2} = @(x,y,z) 1-x;
psi_D1{3} = @(x,y,z) x;
psi_D1{4} = @(x,y,z) -y.*(1-z);
psi_D1{5} = @(x,y,z) -y.*z;
psi_D2 = cell(5,1);
psi_D2{1} = @(x,y,z) -(y-1);
psi_D2{2} = @(x,y,z) y.*(1-z);
psi_D2{3} = @(x,y,z) y.*z;
psi_D2{4} = @(x,y,z) -(1-x);
psi_D2{5} = @(x,y,z) -x;
vpsi1 = zeros(25,27);
vpsi2 = zeros(25,27);
for i = 1:5
    for j = 1:5
        f1 = @(x,y,z) psi_D1{i}(x,y,z).*psi_D1{j}(x,y,z).*y;
        f2 = @(x,y,z) psi_D2{i}(x,y,z).*psi_D2{j}(x,y,z).*y;
        vpsi1( sub2ind([5 5], i , j) , : ) =...
        ( f1( p_cube(:,1) ,p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        vpsi2( sub2ind([5 5], i , j) , : ) =...
        ( f2( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
    end
end

% Create epsi matrices
psi_D1 = cell(3,1);
psi_D1{1} = @(x,y,z) -x.*y;
psi_D1{2} = @(x,y,z) x.*(1-z);
psi_D1{3} = @(x,y,z) x.*z;
psi_D1{4} = @(x,y,z) -x.*(1-y);
psi_D2 = cell(3,1);
psi_D2{1} = @(x,y,z) -x.*y.*z;
psi_D2{2} = @(x,y,z) -x.*(1-y);
psi_D2{3} = @(x,y,z) x;
psi_D2{4} = @(x,y,z) -x.*y.*(1-z);
psi_D3 = cell(3,1);
psi_D3{1} = @(x,y,z) x.*y;
psi_D3{2} = @(x,y,z) -x.*(1-y.*z);
psi_D3{3} = @(x,y,z) x.*(1-y);
psi_D3{4} = @(x,y,z) -x.*y.*z;
psi_D4 = cell(3,1);
psi_D4{1} = @(x,y,z) x.*y.*z;
psi_D4{2} = @(x,y,z) x.*(1-y);
psi_D4{3} = @(x,y,z) x.*y.*(1-z);
psi_D4{4} = @(x,y,z) -x;
psi_D5 = cell(3,1);
psi_D5{1} = @(x,y,z) x.*y.*z;
psi_D5{2} = @(x,y,z) -x.*(1-y);
psi_D5{3} = @(x,y,z) x.*(1-y.*z);
psi_D5{4} = @(x,y,z) -x.*y;
epsi1 = zeros(16,27);
epsi2 = zeros(16,27);
epsi3 = zeros(16,27);
epsi4 = zeros(16,27);
epsi5 = zeros(16,27);
for i = 1:4
    for j = 1:4
        f1 = @(x,y,z) psi_D1{i}(x,y,z).*psi_D1{j}(x,y,z) .*(x.^2);
        f2 = @(x,y,z) psi_D2{i}(x,y,z).*psi_D2{j}(x,y,z) .* (x.^2).*y;
        f3 = @(x,y,z) psi_D3{i}(x,y,z).*psi_D3{j}(x,y,z) .* (x.^2).*y;
        f4 = @(x,y,z) psi_D4{i}(x,y,z).*psi_D4{j}(x,y,z) .* (x.^2).*y;
        f5 = @(x,y,z) psi_D5{i}(x,y,z).*psi_D5{j}(x,y,z) .* (x.^2).*y;
        epsi1( sub2ind([4 4], i , j) , : ) =...
        ( f1( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        epsi2( sub2ind([4 4], i , j) , : ) =...
        ( f2( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        epsi3( sub2ind([4 4], i , j) , : ) =...
        ( f3( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        epsi4( sub2ind([4 4], i , j) , : ) =...
        ( f4( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        epsi5( sub2ind([4 4], i , j) , : ) =...
        ( f5( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
    end
end

% Create tpsi matrices
lambda_D1 = cell(3,1);
lambda_D1{1} = @(z) -z;
lambda_D1{2} = @(z) -(1-z);
lambda_D1{3} = @(z) 1;
lambda_D2 = cell(3,1);
lambda_D2{1} = @(z) -1;
lambda_D2{2} = @(z) (1-z);
lambda_D2{3} = @(z) z;
lambda_D3 = cell(3,1);
lambda_D3{1} = @(z) z;
lambda_D3{2} = @(z) -1;
lambda_D3{3} = @(z) 1-z;
tpsi1 = zeros(9,9);
tpsi2 = zeros(9,9);
tpsi3 = zeros(9,9);
for i = 1:3
    for j = 1:3
        f1 = @(z) lambda_D1{i}(z).*lambda_D1{j}(z);
        f2 = @(z) lambda_D2{i}(z).*lambda_D2{j}(z);
        f3 = @(z) lambda_D3{i}(z).*lambda_D3{j}(z);
        tpsi1( sub2ind([3 3], i , j) , : ) = f1( p_I ).*w_I;
        tpsi2( sub2ind([3 3], i , j) , : ) = f2( p_I ).*w_I;
        tpsi3( sub2ind([3 3], i , j) , : ) = f3( p_I ).*w_I;
    end
end

% Create phi_edge matrices, non overlapping elements

p_I_6 = [
   0.03376524,
   0.16939531,
   0.38069041,
   0.61930959,
   0.83060469,
   0.96623476,
];

w_I_6 = [
    0.08566225,
    0.18038079,
    0.23395697,
    0.23395697,
    0.18038079,
    0.08566225
];

local = cell(1,3);
local{1} = @(x,y) 1-x;
local{2} = @(x,y) x-y;
local{3} = @(x,y) y;
mat_loc = zeros(3);
for i = 1:3
    for j = 1:6
        mat_loc(i,j) = local{i}(p_T_6(j,1),p_T_6(j,2));
    end
end
W = w_T_6*(w_I_6');
disp(W)
N_aux = zeros(18);
phi_edge = zeros(9,36);
for i=1:3
    for j=1:3
        for k = 1:6
            for q=1:6
                N_aux( q + 6*(i-1) , k + 6*(j-1) ) = W(q,k)*mat_loc(i,q)*mat_loc(j,q);
            end
        end
    end
end
for i=1:9
    [im, jm] = ind2sub([3 3] , i);
    im = 6*(im - 1) + 1;
    jm = 6*(jm - 1) + 1;
    phi_edge(i,:) = reshape( N_aux( im:im+5 , jm:jm+5 ) , 1 , [] );
end

% Create epis_edge matrices,
% and ignores the boundary degrees of freedom

epsi_D = cell(3,1);
epsi_D{1} = @(y) y;
epsi_D{2} = @(y) 1;
epsi_D{3} = @(y) y;

epsi1_edge = epsi_D{1}(p_I).^2.*w_I;
epsi2_edge = epsi_D{2}(p_I).^2.*w_I;
epsi3_edge = epsi_D{3}(p_I).^2.*w_I;

% Create vpis_edge matrices, wihtout factor involving s 

vpsi_D1 = cell(3,1);
vpsi_D1{1} = @(x,y,z) 1-x;
vpsi_D1{2} = @(x,y,z) x.*(1-y);
vpsi_D1{3} = @(x,y,z) x.*y;
vpsi_D2 = cell(3,1);
vpsi_D2{1} = @(x,y,z) 1-x.*y;
vpsi_D2{2} = @(x,y,z) x.*y.*(1-z);
vpsi_D2{3} = @(x,y,z) x.*y.*z;

vpsi1_edge = zeros(9,27);
vpsi2_edge = zeros(9,27);
for i = 1:3
    for j = 1:3
        f1 = @(x,y,z) vpsi_D1{i}(x,y,z).*vpsi_D1{j}(x,y,z);
        f2 = @(x,y,z) vpsi_D2{i}(x,y,z).*vpsi_D2{j}(x,y,z).*y;
        vpsi1_edge( sub2ind([3 3], i , j) , : ) =...
        ( f1( p_cube(:,1) ,p_cube(:,2) , p_cube(:,3)) ).*w_cube;
        vpsi2_edge( sub2ind([3 3], i , j) , : ) =...
        ( f2( p_cube(:,1) , p_cube(:,2) , p_cube(:,3)) ).*w_cube;
    end
end

filename = 'data.mat';

save(filename,"p_cube","p_T_12","p_T_6","p_I","w_I", "p_I_6", "w_I_6", ...
    "phiA","phiB","phiD", ...
    "vpsi1","vpsi2", ...
    "epsi1","epsi2","epsi3","epsi4","epsi5", ...
    "tpsi1","tpsi2","tpsi3", ...
    "phi_edge", ...
    "epsi1_edge",'epsi2_edge','epsi3_edge', ...
    "vpsi1_edge","vpsi2_edge")
