function [coefs_mean, frequencies] = count_frequence(signal_405nm,signal_465nm, sampling_rate)

    wave_name = 'cgau8';
    if exist('Wavelet', 'var')
        wave_name = Wavelet;
    end

    total_scale = 128;
    if exist('FrequencyResolution', 'var')
        total_scale = FrequencyResolution;
    end

    processed_signal_405nm = signal_405nm;
    processed_signal_465nm = signal_465nm;

    fitted_405nm = polyfit(processed_signal_405nm, processed_signal_465nm, 1);
    fitted_signal_405nm = polyval(fitted_405nm, processed_signal_405nm);

    delta_f_over_f = ((processed_signal_465nm - fitted_signal_405nm) ./ fitted_signal_405nm) * 100;
    
    cf = centfrq(wave_name);
    scales = 2 * cf * total_scale ./ (total_scale:-1:1);
    [coefs, frequencies] = cwt(delta_f_over_f, scales, wave_name, 1.0 / sampling_rate);
    coefs_mean=median(abs(coefs),2);
end
