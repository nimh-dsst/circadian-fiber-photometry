%% Author Qijun Tang
% make sure giving credit to FPA-master Author

clear all
clc
close all

%% 1.setting parameters
fs=60; % in unit of Hz, 120 Hz is how it is saved when select 100 downsampling when saving. Newest version of the software have 200 downsampling defaul
t_int=1; % in unit of hour. How long is the interval
t_session=610; % in unit of second. How long the session is per interval
t_start='18:05'; % initial time of the recording, in EST
zt_start=14;
date_start='05-Sep-2025'; 
n_Chn=1; % how many channels were recorded (this number is 4 if you have all 4 channels turned on, even you don't have all 4 animals there)
fitting_cutoff=0; %how much percentile on each side to not fit when fitting isos
weight_fit=1; %if need to force the slope to be positive

sortnot=0; % indicate if need to sort the time of the recording sessions, normally should be 0
droplast=0; % sometimes, the last session of some recordings are bad and need to be dropped, then put this to 1 if need to drop the last session

lowpass_not=0; % change this to 0 if don't want to do low-pass filter
cutoff_frequency = 2; % in Hz
lp_filter = designfilt('lowpassfir', 'PassbandFrequency', cutoff_frequency, 'StopbandFrequency', cutoff_frequency + 1, 'PassbandRipple', 1, 'StopbandAttenuation', 60, 'SampleRate', fs);

%

x=1:(t_session-10)*fs; %drop the first and last 5s of data because of the bandpass filter makes it weird
x=transpose(x);
%% 2. read data (from new version of Doric Studio, .doric file)
% after this one, you can run step 7 to quickly visualize the raw data
[filename,path] = uigetfile('*.doric');
cd(path);
[data_org]=readdoric_QT_v4(filename,t_session,n_Chn,fs,sortnot,0);
data_org(:,1)=x;
n_int=size(data_org,3);

data_org(isnan(data_org))=0;
%% 2**. add more recording files (repeat until you load all the files you want)
[filename,path] = uigetfile('*.doric');
cd(path);

data1=data_org;
clear data_org;

[data_org]=readdoric_QT_v4(filename,t_session,n_Chn,fs,sortnot,0);
data_org(:,1)=x;
data2=data_org;
clear data_org;

data_org=cat(3,data1,data2);
clear data1 data2
n_int=size(data_org,3);

%% 2*. a weired config file cause no Analog-out channel in the data, need to use a special "readDoric", run this for the 4 channel config
[filename,path] = uigetfile('*.doric');
cd(path);
[data_org]=readdoric_QT_v4_special(filename,t_session,n_Chn,fs,sortnot,droplast);
data_org(:,1)=x;
n_int=size(data_org,3);

data_org(isnan(data_org))=0;


%% 2**. add more of the wired data recording files (repeat until you load all the files you want)
[filename,path] = uigetfile('*.doric');
cd(path);

data1=data_org;
clear data_org;

[data_org]=readdoric_QT_v4_special(filename,t_session,n_Chn,fs,sortnot,droplast);
data_org(:,1)=x;
data2=data_org;
clear data_org;

data_org=cat(3,data1,data2);
clear data1 data2
n_int=size(data_org,3);
%% 2****. read data_digital (from new version of Doric Studio, .doric file)(have digital IO)
% after this one, you can run step 7 to quickly visualize the raw data
[filename,path] = uigetfile('*.doric');
cd(path);
[data_org]=readdoric_QT_v4_digital(filename,t_session,n_Chn,fs,sortnot,0);
data_org(:,1)=x;
n_int=size(data_org,3);

data_org(isnan(data_org))=0;
%% 2****. add more recording files (repeat until you load all the files you want)
[filename,path] = uigetfile('*.doric');
cd(path);

data1=data_org;
clear data_org;

[data_org]=readdoric_QT_v4_digital(filename,t_session,n_Chn,fs,sortnot,0);
data_org(:,1)=x;
data2=data_org;
clear data_org;

data_org=cat(3,data1,data2);
clear data1 data2
n_int=size(data_org,3);

%% only to save my and MY's ass
[filename_405,path] = uigetfile('*.doric');
cd(path);
[data_405]=readdoric_QT_v5_fix_corrupt(filename_405,t_session,1,fs,sortnot,droplast);

[filename_465,path] = uigetfile('*.doric');
[data_465]=readdoric_QT_v5_fix_corrupt(filename_465,t_session,1,fs,sortnot,droplast);

data_org2=data_405;
for n=1:size(data_465,3)
    data_org2(:,3,n)=data_465(:,2,n);
    data_org2(:,4,n)=data_405(:,2,n);
    data_org2(:,5,n)=data_465(:,2,n);
end
data_org=cat(3,data_org,data_org2);
n_int=size(data_org,3);

%% 3. fit 405 to 465 to calculate df/f
for i=1:n_Chn
    % 405 should be fitted onto 465 on the entire recording
    all_405=reshape(data_org(:,i*2,:), [], 1);
    all_465=reshape(data_org(:,i*2+1,:), [], 1);
    matix_cutfit=all_465>prctile(all_465,fitting_cutoff) & all_465<prctile(all_465,100-fitting_cutoff);%not fitting the date points outside of cutoff
    if weight_fit
        all_405(1:t_session*fs)=0;
        all_465(1:t_session*fs)=0;
    end
    fit_all(i,:)=polyfit(all_405(matix_cutfit),all_465(matix_cutfit),1);
    beta_prev=[];
    for n=1:n_int
        y_405=data_org(:,i*2,n); 
        y_465=data_org(:,i*2+1,n);

        fit_temp=fit_all(i,1)*y_405+fit_all(i,2);
        fit_405_curve(:,i,n)=movmean(fit_temp,5*fs);
        data_dff(:,i,n)=(y_465-fit_405_curve(:,i,n))./median(fit_405_curve(:,i,n));
        % further dynamic fit to correct artifacts
        y_dff_temp=data_dff(:,i,n);
        y_isos_temp=y_405;
        [data_dff_dynamiccorrected(:,i,n),beta_prev,fitted_dynamiccorrected(:,i,n),calcium_filt(:,i,n)]=FP_IRLS_regularized_v9(y_dff_temp, y_isos_temp, fs, 2.5*60*fs, 0.6, 4.685,beta_prev);
        beta_prev=[];
        data_dff_2padj(:,i,n)=data_dff_dynamiccorrected(:,i,n)-prctile(data_dff_dynamiccorrected(:,i,n), 10);
        [coefs_mean(:,i,n), frequencies(:,i,n)] = count_frequence(y_405,y_465,fs);
    end
end

% cut off 2% of dff

data_dff_cut0=data_dff_2padj;
data_dff_cut0(data_dff_cut0<0)=0;

%count events using the curve above the cutoff line
for i=1:n_Chn
    for n=1:n_int
        % events(i,n)=count_event_2pc(data_dff_cut0(:,i,n),fs);
        events(i,n)=count_event_2pc_v9(data_dff_dynamiccorrected(:,i,n),fs);
    end
end



%% 4. plot processed data

for n=1:n_int
    for i=1:n_Chn
        level_phasic(i,n)=sum(data_dff_cut0(:,i,n));
        level_cal_raw(i,n)=median(data_org(:,i*2+1,n));
        level_405_raw(i,n)=median(data_org(:,i*2,n));
        level_tonic(i,n)=prctile(data_dff(:,i,n),10); %because 2% is used to cut off, it will be the tonic part
        level_average(i,n)=median(data_dff(:,i,n));
    end
end

level_tonic_dtr=level_tonic-movmean(level_tonic,24/t_int,2);
level_tonic_z=(level_tonic-movmean(level_tonic,24/t_int,2))./movstd(level_tonic,24/t_int,0,2);
level_average_dtr=level_average-movmean(level_average,24/t_int,2);
% level_tonic_z=(level_tonic-movmean(level_tonic,24/t_int,2))./std(level_tonic); %since only devided by a same number, this is equivilant to the level_tonic_dtr above

figure
for i=1:n_Chn
    t=1:n_int;
    subplot(8,n_Chn,i)
    % plot(level_cal_ratio(i,:));
    plot(t,level_405_raw(i,:));
    xlim([1 n_int]);
    title('average raw 405')

    subplot(8,n_Chn,i+n_Chn)
    plot(t,level_cal_raw(i,:));
    xlim([1 n_int]);
    title('average raw 465')

    subplot(8,n_Chn,i+2*n_Chn)
    plot(t,level_tonic(i,:));
    xlim([1 n_int]);
    title('Tonic (baseline)')

    subplot(8,n_Chn,i+3*n_Chn)
    plot(t,level_tonic_dtr(i,:));
    xlim([1 n_int]);
    title('Tonic (detrended by 24 movemean)')

    subplot(8,n_Chn,i+4*n_Chn)
    plot(t,level_tonic_z(i,:));
    xlim([1 n_int]);
    title('Tonic (z-score to 24 movestd)')

    subplot(8,n_Chn,i+5*n_Chn)
    plot(t,level_phasic(i,:));
    xlim([1 n_int]);
    title('Phasic (transients)')

    subplot(8,n_Chn,i+6*n_Chn)
    plot(t,events(i,:));
    xlim([1 n_int]);
    title('# of events')

    subplot(8,n_Chn,i+7*n_Chn)
    z_lim = -0.02:0.01:1.0;
    contourf(t, frequencies(:,1,1), squeeze(coefs_mean(:,i,:)), z_lim, 'LineStyle', 'none');
    title('power map')
end

%% 5.save the processed data
save('data_all_proc_v9','events','data_org','n_Chn','n_int','level_tonic','level_phasic','level_tonic_dtr','data_dff_2padj','data_dff','data_dff_cut0',"fit_405_curve","level_tonic_z","fitting_cutoff","weight_fit","coefs_mean","frequencies","level_average","level_average_dtr");


 %% 6.export data for clocklab analysis
for n=1:n_Chn
table_AUC=cell(n_int+7,1); %there is a 7-row header
%make header
table_AUC(1)={['v9Phasic-Chn',num2str(n)]};
table_AUC(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_AUC(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_AUC(4)={t_int*60*4}; % interval time (in minute) times 4
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_AUC(6)={'life sucks'}; % place holder that you can put info
table_AUC(7)={'male'}; % place holder that you can put info
table_AUC(8:end)=num2cell(level_phasic(n,:));
% export
filename_AUC=[table_AUC{1},'.awd'];
writecell(table_AUC,filename_AUC,'FileType','text');
end

for n=1:n_Chn
table_cal=cell(n_int+7,1); %there is a 7-row header
%make header
table_cal(1)={['v9Tonic-Chn',num2str(n)]};
table_cal(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_cal(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_cal(4)={t_int*60*4}; % number of intervals per day times 10
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_cal(6)={'life sucks'}; % place holder that you can put info
table_cal(7)={'male'}; % place holder that you can put info
table_cal(8:end)=num2cell(level_tonic(n,:)-min(level_tonic(n,:)));
% export
filename_AUC=[table_cal{1},'.awd'];
writecell(table_cal,filename_AUC,'FileType','text');
end

for n=1:n_Chn
table_cal=cell(n_int+7,1); %there is a 7-row header
%make header
table_cal(1)={['v9Tonic_dtr-Chn',num2str(n)]};
table_cal(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_cal(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_cal(4)={t_int*60*4}; % number of intervals per day times 10
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_cal(6)={'life sucks'}; % place holder that you can put info
table_cal(7)={'male'}; % place holder that you can put info
table_cal(8:end)=num2cell(level_tonic_dtr(n,:));
% export
filename_AUC=[table_cal{1},'.awd'];
writecell(table_cal,filename_AUC,'FileType','text');
end

for n=1:n_Chn
table_cal=cell(n_int+7,1); %there is a 7-row header
%make header
table_cal(1)={['v9Events-Chn',num2str(n)]};
table_cal(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_cal(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_cal(4)={t_int*60*4}; % number of intervals per day times 10
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_cal(6)={'life sucks'}; % place holder that you can put info
table_cal(7)={'male'}; % place holder that you can put info
table_cal(8:end)=num2cell(events(n,:));
% export
filename_AUC=[table_cal{1},'.awd'];
writecell(table_cal,filename_AUC,'FileType','text');
end

for n=1:n_Chn
table_mean=cell(n_int+7,1); %there is a 7-row header
%make header
table_mean(1)={['v9Average-Chn',num2str(n)]};
table_mean(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_mean(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_mean(4)={t_int*60*4}; % number of intervals per day times 10
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_mean(6)={'life sucks'}; % place holder that you can put info
table_mean(7)={'male'}; % place holder that you can put info
table_mean(8:end)=num2cell(level_average(n,:));
% export
filename_AUC=[table_mean{1},'.awd'];
writecell(table_mean,filename_AUC,'FileType','text');
end

for n=1:n_Chn
table_mean=cell(n_int+7,1); %there is a 7-row header
%make header
table_mean(1)={['v9Average_dtr-Chn',num2str(n)]};
table_mean(2)={date_start}; % just put a 2020-01-01 as initial date, real info refer to lab notes
table_mean(3)={t_start}; % default to put 12:00 as initial time, real info refer to lab notes
table_mean(4)={t_int*60*4}; % number of intervals per day times 10
table_AUC(5)={['ZT-',num2str(zt_start)]}; % just place holder
table_mean(6)={'life sucks'}; % place holder that you can put info
table_mean(7)={'male'}; % place holder that you can put info
table_mean(8:end)=num2cell(level_average_dtr(n,:));
% export
filename_AUC=[table_mean{1},'.awd'];
writecell(table_mean,filename_AUC,'FileType','text');
end



%% 7. show raw data (this step is just to visualize raw data, no need to run if you don't need to)
figure
mouse=1;
startday=1;  % day to start for raw trace, default is 1 which start from beginning
day=1;  % day to look at for the individual traces
rate_downsample=1;
length_session=(t_session-10)*fs;
x=1:rate_downsample:length_session;
y=zeros(length_session/1)+1;
hold on
for n=(startday-1)*24+1:size(data_org,3)
    plot(length_session*(n-1)+x,data_org(x,2*mouse,n));
    plot(length_session*(n-1)+x,data_org(x,2*mouse+1,n));
    % plot(length_session*(n-1)+x,fit_405_curve(x,mouse,n));
    % plot(length_session*(n-1)+x,data_dff(x,mouse,n));
end
title('Whole recording')

figure
% spacing=max(max(max(data_dff_2padj)));
spacing=0.2;
subplot(1,2,1)
for n=12-zt_start+1+24*(day-1):24-zt_start+24*(day-1)
    m=n-(12-zt_start+1+24*(day-1));
    temp=data_dff_2padj(:,mouse,n)-(12+spacing*m);
    plot(downsample(temp,60),'g');
    hold on
end
title('Dark phase')

subplot(1,2,2)
for n=24-zt_start+1+24*(day-1):36-zt_start+24*(day-1)
    m=n-(24-zt_start+1+24*(day-1));
    temp=data_dff_2padj(:,mouse,n)-(spacing*m);
    plot(downsample(temp,60),'g');
    hold on
end
title('Light phase')

%% stack traces for correction too
figure
% spacing=max(max(max(data_dff_2padj)));
spacing=0.2;
subplot(1,2,1)
for n=12-zt_start+1+24*(day-1):24-zt_start+24*(day-1)
    m=n-(12-zt_start+1+24*(day-1));
    temp1=calcium_filt(:,mouse,n)-(12+spacing*m);
    temp2=fitted_dynamiccorrected(:,mouse,n)-(12+spacing*m);
    plot(downsample(temp1,60),'g');
    hold on
    plot(downsample(temp2,60),'k');
end
title('Dark phase')

subplot(1,2,2)
for n=24-zt_start+1+24*(day-1):36-zt_start+24*(day-1)
    m=n-(24-zt_start+1+24*(day-1));
    temp1=calcium_filt(:,mouse,n)-(spacing*m);
    temp2=fitted_dynamiccorrected(:,mouse,n)-(spacing*m);
    plot(downsample(temp1,60),'g');
    hold on
    plot(downsample(temp2,60),'k');
end
title('Light phase')

%% 8. light response data
animal=1;
int=1;
rate_downsample=10;
windows=[115*fs/rate_downsample:130*fs/rate_downsample;235*fs/rate_downsample:250*fs/rate_downsample;355*fs/rate_downsample:370*fs/rate_downsample];
y = zeros(size(windows))+1;

figure 
hold on
plot(data_org(1:rate_downsample:end,animal*2,int));
plot(data_org(1:rate_downsample:end,animal*2+1,int));
plot(data_dff(1:rate_downsample:end,animal,int));
for n=1:3
    plot(windows(n,:),y(n,:));
end
%% 9. processing light response data (window)
int_beforeCT6LP=63;
for n=1:n_Chn
ct6_dff=data_dff(:,n,int_beforeCT6LP+1);
ct14_dff=data_dff(:,n,int_beforeCT6LP+1+8);
ct22_dff=data_dff(:,n,int_beforeCT6LP+1+8+8);

figure
hold on
plot(ct6_dff);
plot(ct14_dff);
plot(ct22_dff);

% downsample and extract window
ct6_dff_down=downsample(ct6_dff,fs);
ct14_dff_down=downsample(ct14_dff,fs);
ct22_dff_down=downsample(ct22_dff,fs);
window1_6 = ct6_dff_down(101:190);
window2_6 = ct6_dff_down(221:310);
window3_6 = ct6_dff_down(341:430);
ct6_dff_windows = [window1_6, window2_6, window3_6];
for i=1:size(ct6_dff_windows,2)
    ct6_dff_windows_shifted(:,i+(n-1)*3)=ct6_dff_windows(:,i)-mean(ct6_dff_windows(1:15,i));
    ct6_dff_windows_z(:,i+(n-1)*3)=(ct6_dff_windows(:,i)-mean(ct6_dff_windows(1:15,i)))./std(ct6_dff_windows(1:15,i));
end
window1_14 = ct14_dff_down(101:190);
window2_14 = ct14_dff_down(221:310);
window3_14 = ct14_dff_down(341:430);
ct14_dff_windows = [window1_14, window2_14, window3_14];
for i=1:size(ct14_dff_windows,2)
    ct14_dff_windows_shifted(:,i+(n-1)*3)=ct14_dff_windows(:,i)-mean(ct14_dff_windows(1:15,i));
    ct14_dff_windows_z(:,i+(n-1)*3)=(ct14_dff_windows(:,i)-mean(ct14_dff_windows(1:15,i)))./std(ct14_dff_windows(1:15,i));
end
window1_22 = ct22_dff_down(101:190);
window2_22 = ct22_dff_down(221:310);
window3_22 = ct22_dff_down(341:430);
ct22_dff_windows = [window1_22, window2_22, window3_22];
for i=1:size(ct22_dff_windows,2)
    ct22_dff_windows_shifted(:,i+(n-1)*3)=ct22_dff_windows(:,i)-mean(ct22_dff_windows(1:15,i));
    ct22_dff_windows_z(:,i+(n-1)*3)=(ct22_dff_windows(:,i)-mean(ct22_dff_windows(1:15,i)))./std(ct22_dff_windows(1:15,i));
end
end
% calculate AUC
ct6_auc(1,:)=trapz(ct6_dff_windows_shifted(1:15,:));
ct14_auc(1,:)=trapz(ct14_dff_windows_shifted(1:15,:));
ct22_auc(1,:)=trapz(ct22_dff_windows_shifted(1:15,:));

ct6_auc(2,:)=trapz(ct6_dff_windows_shifted(16:30,:));
ct14_auc(2,:)=trapz(ct14_dff_windows_shifted(16:30,:));
ct22_auc(2,:)=trapz(ct22_dff_windows_shifted(16:30,:));

ct6_auc(3,:)=trapz(ct6_dff_windows_shifted(31:90,:));
ct14_auc(3,:)=trapz(ct14_dff_windows_shifted(31:90,:));
ct22_auc(3,:)=trapz(ct22_dff_windows_shifted(31:90,:));


%save('lightpulse_dff','ct6_dff','ct14_dff',"ct22_dff");

%% random extra stuff
animal=1;
int=66;
rate_downsample=10;
windows=[115*fs/rate_downsample:130*fs/rate_downsample;235*fs/rate_downsample:250*fs/rate_downsample;355*fs/rate_downsample:370*fs/rate_downsample];
y = zeros(size(windows));
figure
hold on
plot(data_org(1:rate_downsample:end,animal*2,int));
plot(data_org(1:rate_downsample:end,animal*2+1,int));
plot(data_dff(1:rate_downsample:end,animal,int));
for n=1:3
    plot(windows(n,:),y(n,:));
end
%% save the data
save('lightpulse_dff','ct6_dff','ct14_dff');
%% for short term
save(filename(1:end-6));
ct6_dff=data_dff;
ct14_dff=data_dff;


%% moving std
windowsize=15; %in unit of s
ct14_std=movstd(ct14_dff,windowsize*fs/2);
ct14_std_down=rate_downsample(ct14_std,fs);

