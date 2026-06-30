%% This is for reading doric files to be used in QT fiber photometry processing code 
function [data_org]=readdoric_QT(filename,t_session,n_Chn,samplerate,sortnot,droplast)
% Example on how to read a .doric file using Matlab
%
% Note:
% This example will not work with Octave. In case you want to use Octave you can
% use the load() function and you can navigate in the struct with the help of
% some functions like isstruct(), fieldnames (), numfields()
%
% Example file generated with an FPC
% There is two way to do it Automatically or Manually


%First set the name of the file that you want to extract the data from in
%this example the file is in the current folder where we have the matlab script
%filename = 'Console_Acq_0000.doric'; filename passed through function
%already

Data_Acquired = ExtractDataAcquisition(filename);
%%
%--------------------Manually-----------------------------
% this is to load all the data that looks like old way in cvs so compatable
% with old scripts
% TimeIn = h5read(filename,'/DataAcquisition/FPConsole/Signals/Series0001/AnalogIn/Time');
% [dim1,dim2]=size(TimeIn);
% data=zeros(dim1,1+length(Data_Acquired)/2*3);
% data(:,1)=TimeIn;
n_int=length(Data_Acquired)/(n_Chn*2+3);

if sortnot
for n=1:n_int-droplast
    t(n)=Data_Acquired((n_Chn*2+3)*(n-1)+1).Data(1).Data(1);
end

[t_sorted,index]=sort(t);

for m=1:n_int-droplast
    n=index(m);
    data_org(:,1,m)= Data_Acquired((n_Chn*2+3)*(n-1)+1).Data(1).Data((samplerate*5)+1:(t_session-5)*samplerate);
   for i=1:n_Chn
       data_org(:,i*2,m)= Data_Acquired((n_Chn*2+3)*(n-1)+2*i-1).Data(2).Data((samplerate*5)+1:(t_session-5)*samplerate);
       data_org(:,i*2+1,m)=Data_Acquired((n_Chn*2+3)*(n-1)+2*i).Data(2).Data((samplerate*5)+1:(t_session-5)*samplerate);
   end
end
else
for m=1:n_int-droplast
    n=m;
    data_org(:,1,m)= Data_Acquired((n_Chn*2+3)*(n-1)+1).Data(1).Data((samplerate*5)+1:(t_session-5)*samplerate);
   for i=1:n_Chn
       data_org(:,i*2,m)= Data_Acquired((n_Chn*2+3)*(n-1)+2*i-1).Data(2).Data((samplerate*5)+1:(t_session-5)*samplerate);
       data_org(:,i*2+1,m)=Data_Acquired((n_Chn*2+3)*(n-1)+2*i).Data(2).Data((samplerate*5)+1:(t_session-5)*samplerate);
   end
end 
end


