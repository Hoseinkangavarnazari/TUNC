set(groot, 'defaultFigureCloseRequestFcn', 'close(gcf)');   % avoid R2023b crash after closing figures


%plot_3d_for_dpa_fix();
%plot_3d_for_dpa_random();
%plot_3d_for_tpa_random();

%compareCFFandGC();
mhdEvaluation();
%numberOfKeys();

function plot_3d_for_dpa_fix()
    runs = 100000;
    max_subset_size = 9;
    max_hop = 8;

    success_ratio_matrix = zeros(max_subset_size, max_hop);

    for hop = 1:max_hop
        for keys = 1:max_subset_size
            filename = sprintf('./dpa_fix/results_%druns_%dkeys_%dhops.csv', runs, keys, hop);
            data = readtable(filename);
            success_count = sum(strcmp(data.is_success, 'True'));
            success_ratios = success_count / runs;

            success_ratio_matrix(keys, hop) = success_ratios;
        end
    end

    bar3(success_ratio_matrix);

    ax = gca;
    set(ax, 'ZScale', 'log');

    xlabel('Hop');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs / Relay Node Keyset Size / hop\n(Max Hops = %d, Key Pool Size = %d)', runs, max_hop, max_subset_size + 1));
    view(69, 23); % azimuth, elevation

    % 调整 Z 轴数据以适应对数刻度
    zlim = 10^(-10);
    h = get(gca, 'Children');
    for i = 1:length(h)
        zdata = get(h(i), 'ZData');
        zdata(zdata < zlim) = zlim;
        set(h(i), 'ZData', zdata);
    end

    fig = gcf;
    fig.Position = [0, 0, 800, 600];
    print(fig, '-dpng', sprintf('./dpa_fix/plot_3d_%druns.png', runs), '-r00');
end


function compareCFFandGC()
    m_runs = 10^6;
    m_max_compromised_nodes = 8;
    m_max_keyset_size = 9;

    m_folder_wp = '/Users/xingyuzhou/Downloads/dpa_wp_our_106_22/merged';
    success_ratio_vector_wp = zeros(1, m_max_compromised_nodes);
    keypool_size_vector_wp = zeros(1, m_max_compromised_nodes);
    keyset_size_vector_wp = zeros(1, m_max_compromised_nodes);
    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_wp_our_%druns_%dc.csv', m_folder_wp, m_runs, compromised_nodes);
        data = readtable(filename);
        success_ratio_vector_wp(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        keypool_size_vector_wp(compromised_nodes) = mean(data.keypool_size);
        keyset_size_vector_wp(compromised_nodes) = mean(data.avg_keyset_size);
    end

    m_folder_greedy = '/Users/xingyuzhou/Downloads/dpa_greedy_our_106_22/merged';
    success_ratio_vector_greedy = zeros(1, m_max_compromised_nodes);
    keypool_size_vector_greedy = zeros(1, m_max_compromised_nodes);
    keyset_size_vector_greedy = zeros(1, m_max_compromised_nodes);
    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_greedy_our_%druns_%dc.csv', m_folder_greedy, m_runs, compromised_nodes);
        data = readtable(filename);
        success_ratio_vector_greedy(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        keypool_size_vector_greedy(compromised_nodes) = mean(data.keypool_size);
        keyset_size_vector_greedy(compromised_nodes) = mean(data.avg_keyset_size);
    end

    m_folder_cff = '/Users/xingyuzhou/Downloads/dpa_cff_our_106_22/merged';
    success_ratio_vector_cff = zeros(1, m_max_compromised_nodes);
    keypool_size_vector_cff = zeros(1, m_max_compromised_nodes);
    keyset_size_vector_cff = zeros(1, m_max_compromised_nodes);
    data_count_checks = cell(1, m_max_compromised_nodes);
    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_cff_our_%druns_%dc.csv', m_folder_cff, m_runs, compromised_nodes);
        data = readtable(filename);

        success_ratio_vector_cff(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        keypool_size_vector_cff(compromised_nodes) = mean(data.keypool_size);
        keyset_size_vector_cff(compromised_nodes) = mean(data.avg_keyset_size);
        data_count_checks{compromised_nodes} = data.count_checks;
    end

    m_folder_mhd = '/Users/xingyuzhou/Downloads/dpa_mhd_our_456_106_22/merged';
    success_ratio_matrix_mhd = zeros(m_max_keyset_size, m_max_compromised_nodes);
    keypool_size_matrix_mhd = zeros(m_max_keyset_size, m_max_compromised_nodes);
    keyset_size_matrix_mhd = zeros(m_max_keyset_size, m_max_compromised_nodes);
    for compromised_nodes = 1:1:m_max_compromised_nodes
        for keyset_size = [4, 5, 6]
            filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd, m_runs, compromised_nodes, keyset_size);
            data = readtable(filename);
            success_ratio_matrix_mhd(keyset_size, compromised_nodes) = sum(data.is_success == 1) / m_runs;
            keypool_size_matrix_mhd(keyset_size, compromised_nodes) = 10;
            keyset_size_matrix_mhd(keyset_size, compromised_nodes) = 5;
        end
    end

    m_folder_mhd_2 = '/Users/xingyuzhou/Downloads/dpa_mhd_our_5_106_22/merged';
    success_ratio_vector_mhd_2 = zeros(1, m_max_compromised_nodes);
    keypool_size_vector_mhd_2 = zeros(1, m_max_compromised_nodes);
    keyset_size_vector_mhd_2 = zeros(1, m_max_compromised_nodes);
    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd_2, m_runs, compromised_nodes, 5);
        data = readtable(filename);
        success_ratio_vector_mhd_2(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        keypool_size_vector_mhd_2(compromised_nodes) = 11;
        keyset_size_vector_mhd_2(compromised_nodes) = 5;
    end

    m_folder_mhd_4 = '/Users/xingyuzhou/Downloads/dpa_mhd_our_106/merged';
    success_ratio_matrix_mhd_4 = zeros(6, 5);
    for compromised_nodes = 1:2:10
        for keyset_size = 1:2:11
            filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd_4, m_runs, compromised_nodes, keyset_size);
            data = readtable(filename);
            success_ratio_matrix_mhd_4((keyset_size + 1) / 2, (compromised_nodes + 1) / 2) = sum(data.is_success == 1) / m_runs;
        end
    end


%    security_performance_comparison
%    fig1 = figure();
%    plot(1:m_max_compromised_nodes, success_ratio_vector_cff, '-o', 'DisplayName', 'CFF-based', 'LineWidth', 2);
%    hold on;
%    plot(1:m_max_compromised_nodes, success_ratio_matrix_mhd(5, :), '-x', 'DisplayName', 'Max-Hamming PKA (Config 1)', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, success_ratio_vector_mhd_2, '-+', 'DisplayName', 'Max-Hamming PKA (Config 2)', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, success_ratio_vector_greedy, '-s', 'DisplayName', 'Greedy MKA', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, success_ratio_vector_wp, '-d', 'DisplayName', 'Welsh-Powell MKA', 'LineWidth', 2);
%    hold off;
%    set(gca(), 'YScale', 'log');
%    adjustYAxisForLogScale(m_runs);
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Success Probability', 'FontName', 'Open Sans', 'FontSize', 16);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);
%    legend('Location', 'best', 'FontName', 'Open Sans', 'FontSize', 14);


%    average_keypool_size_comparison
%    fig4 = figure();
%    plot(1:m_max_compromised_nodes, keypool_size_vector_cff, '-o', 'DisplayName', 'CFF-based', 'LineWidth', 2);
%    hold on;
%    plot(1:m_max_compromised_nodes, keypool_size_matrix_mhd(5, :), '-x', 'DisplayName', 'Max-Hamming PKA (Config 1)', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, keypool_size_vector_mhd_2, '-+', 'DisplayName', 'Max-Hamming PKA (Config 2)', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, keypool_size_vector_greedy, '-s', 'DisplayName', 'Greedy MKA', 'LineWidth', 2);
%    plot(1:m_max_compromised_nodes, keypool_size_vector_wp, '-d', 'DisplayName', 'Welsh-Powell MKA', 'LineWidth', 2);
%    hold off;
%    set(gca(), 'YScale', 'log');
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Average Keypool Size', 'FontName', 'Open Sans', 'FontSize', 16);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);
%    legend('Location', 'best', 'FontName', 'Open Sans', 'FontSize', 14);


%    average_keyset_size_comparison
%    fig5 = figure();
%    barWidth = 0.15; % 条形的宽度
%    bar((1:m_max_compromised_nodes) - barWidth * 2, keyset_size_vector_cff, barWidth, 'DisplayName', 'CFF-based');
%    hold on;
%    bar((1:m_max_compromised_nodes) - barWidth, keyset_size_matrix_mhd(5, :), barWidth, 'DisplayName', 'Max-Hamming PKA (Config 1)');
%    bar(1:m_max_compromised_nodes, keyset_size_vector_mhd_2, barWidth, 'DisplayName', 'Max-Hamming PKA (Config 2)');
%    bar((1:m_max_compromised_nodes) + barWidth, keyset_size_vector_greedy, barWidth, 'DisplayName', 'Greedy MKA');
%    bar((1:m_max_compromised_nodes) + barWidth * 2, keyset_size_vector_wp, barWidth, 'DisplayName', 'Welsh-Powell MKA');
%    hold off;
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Average Keyset Size', 'FontName', 'Open Sans', 'FontSize', 16);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);
%    legend('Location', 'best', 'FontName', 'Open Sans', 'FontSize', 14);


%    check_count_distribution
%    fig2 = figure();
%    boxplot(cell2mat(data_count_checks), 'Labels', 1:m_max_compromised_nodes, 'Symbol', '', 'Whisker', 1.5);
%    lines = findobj(gca, 'type', 'line', 'Tag', 'Box');
%    medianlines = findobj(gca, 'type', 'line', 'Tag', 'Median');
%    whiskerlines = findobj(gca, 'LineStyle', '--');
%    set(lines, 'LineWidth', 2);
%    set(medianlines, 'LineWidth', 2);
%    set(whiskerlines, 'LineWidth', 2, 'Color', hex2rgb('#00305e'), 'LineStyle', '-');
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Checks Count', 'FontName', 'Open Sans', 'FontSize', 16);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);


%    max_hamming_p28
    fig3 = figure(3);
    bar3(success_ratio_matrix_mhd_4);
    set(gca, 'XTickLabel', {'1', '3', '5', '7', '9'});
    set(gca, 'YTickLabel', {'1', '3', '5', '7', '9', '11'});
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
    ylabel('Relay Node Keyset Size', 'FontName', 'Open Sans', 'FontSize', 16);
    zlabel('Success Probability', 'FontName', 'Open Sans', 'FontSize', 16);
    adjustZAxisForLogScale(m_runs);
end


function mhdEvaluation()
    m_folder_mhd_2 = '/Users/xingyuzhou/Downloads/dpa_mhd_our_5_106_22/merged';
    m_folder_mhd_3 = '/Users/xingyuzhou/Downloads/dpa_mhd_our_2c_106_22/merged';
    m_max_keyset_size_2 = 10;
    m_runs = 10^6;
    m_max_compromised_nodes = 8;

    success_ratio_vector_mhd_2 = zeros(1, m_max_compromised_nodes);
    success_ratio_vector_mhd_3 = zeros(1, m_max_keyset_size_2);
    success_ratio_matrix = NaN(m_max_keyset_size_2, m_max_compromised_nodes);

    for keyset_size = 1:1:m_max_keyset_size_2
        filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd_3, m_runs, 2, keyset_size);
        data = readtable(filename);
        success_ratio_vector_mhd_3(keyset_size) = sum(data.is_success == 1) / m_runs;
        success_ratio_matrix(keyset_size, 2) = sum(data.is_success == 1) / m_runs;
    end

    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd_2, m_runs, compromised_nodes, 5);
        data = readtable(filename);
        success_ratio_vector_mhd_2(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        success_ratio_matrix(5, compromised_nodes) = sum(data.is_success == 1) / m_runs;
    end


%    fig1 = figure();
%    subplot(1, 2, 1);
%    plot(1:m_max_compromised_nodes, success_ratio_vector_mhd_2, '-+', 'DisplayName', 'GC with Keyset Size 5, Keypool Size 11', 'LineWidth', 2, 'Color', hex2rgb('#e1aa29'));
%    set(gca(), 'YScale', 'log');
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Success Probability', 'FontName', 'Open Sans', 'FontSize', 16);
%    adjustYAxisForLogScale(m_runs);
%    title(sprintf('Success Probability vs. Compromised Nodes'), 'FontName', 'Open Sans', 'FontSize', 20);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);
%    legend('Location', 'best', 'FontName', 'Open Sans', 'FontSize', 14);
%    subplot(1, 2, 2)
%    plot(1:m_max_keyset_size_2, success_ratio_vector_mhd_3, '-o', 'DisplayName', 'GC with Keypool Size 11 under Compromised Nodes 2', 'LineWidth', 2);
%    set(gca(), 'YScale', 'log');
%    xlabel('Keyset Size', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Success Probability', 'FontName', 'Open Sans', 'FontSize', 16);
%    adjustYAxisForLogScale(m_runs);
%    title(sprintf('Success Probability vs. Keyset Size'), 'FontName', 'Open Sans', 'FontSize', 20);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);
%    legend('Location', 'best', 'FontName', 'Open Sans', 'FontSize', 14);

%    fig2 = figure();
%    bar3(success_ratio_matrix);
%    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
%    ylabel('Relay Node Keyset Size', 'FontName', 'Open Sans', 'FontSize', 16);
%    zlabel('Success Probability', 'FontName', 'Open Sans', 'FontSize', 16);
%    set(gca(), 'ZScale', 'log');
%    adjustZAxisForLogScale(m_runs);
%    grid on;
%    set(gca, 'GridLineStyle', '--', 'GridColor', hex2rgb('#727879'), 'GridAlpha', 0.4, 'LineWidth', 1);

    disp(success_ratio_matrix);

end



function [] = numberOfKeys()
    w = 0:8; % w 的值域
    q = 10^-3;
    n = 10;
    e = exp(1); % 自然常数 e

    y1 = e * (w+1) .* log(1/q);
    y2 = e * log(1/q) * ones(size(w));
    y3 = e * (w+1).^2 * log(n);
    y4 = e * (w+1) * log(n);
    y5 = 11 * ones(size(w));
    y6 = 5 * ones(size(w));

    figure;
    plot(w, y1, 'LineWidth', 3, 'Color', hex2rgb('#009ee0'));
    hold on;
    plot(w, y2, 'LineWidth', 3, 'Color', hex2rgb('#cd296a'));
%    plot(w, y3, 'LineWidth', 3, 'Color', hex2rgb('#009ee0'), 'LineStyle', '--');
%    plot(w, y4, 'LineWidth', 3, 'Color', hex2rgb('#cd296a'), 'LineStyle', '--');
    plot(w, y5, 'LineWidth', 3, 'Color', hex2rgb('#009ee0'), 'LineStyle', '-.');
    plot(w, y6, 'LineWidth', 3, 'Color', hex2rgb('#cd296a'), 'LineStyle', '-.');

    plot(w, y1, 'o', 'MarkerEdgeColor', hex2rgb('#009ee0'), 'MarkerFaceColor', hex2rgb('#009ee0'), 'MarkerSize', 10);
    plot(w, y2, 'o', 'MarkerEdgeColor', hex2rgb('#cd296a'), 'MarkerFaceColor', hex2rgb('#cd296a'), 'MarkerSize', 10);
%    plot(w, y3, 's', 'MarkerEdgeColor', hex2rgb('#009ee0'), 'MarkerSize', 10);
%    plot(w, y4, 's', 'MarkerEdgeColor', hex2rgb('#cd296a'), 'MarkerSize', 10);
    plot(w, y5, 's', 'MarkerEdgeColor', hex2rgb('#009ee0'), 'MarkerFaceColor', hex2rgb('#009ee0'), 'MarkerSize', 10);
    plot(w, y6, 's', 'MarkerEdgeColor', hex2rgb('#cd296a'), 'MarkerFaceColor', hex2rgb('#cd296a'), 'MarkerSize', 10);

    w_index = find(w == 8);
    if ~isempty(w_index)
        text(8, y1(w_index), ['\leftarrow ' num2str(ceil(y1(w_index)), '%d')], 'Color', hex2rgb('#009ee0'), 'FontSize', 18);
        text(8, y2(w_index), ['\leftarrow ' num2str(ceil(y2(w_index)), '%d')], 'Color', hex2rgb('#cd296a'), 'FontSize', 18);
%        text(8, y3(w_index), ['\leftarrow ' num2str(ceil(y3(w_index)), '%d')], 'Color', hex2rgb('#009ee0'), 'FontSize', 18);
%        text(8, y4(w_index), ['\leftarrow ' num2str(ceil(y4(w_index)), '%d')], 'Color', hex2rgb('#cd296a'), 'FontSize', 18);
        text(8, y5(w_index), ['\leftarrow ' num2str(ceil(y5(w_index)), '%d')], 'Color', hex2rgb('#009ee0'), 'FontSize', 18);
        text(8, y6(w_index), ['\leftarrow ' num2str(ceil(y6(w_index)), '%d')], 'Color', hex2rgb('#cd296a'), 'FontSize', 18);
    end

    hold off;

    xlabel('Compromised Nodes', 'FontName', 'Open Sans', 'FontSize', 16);
    ylabel('Number of Keys', 'FontName', 'Open Sans', 'FontSize', 16);
    title('Keypool Size and Keyset Size Comparison', 'FontName', 'Open Sans', 'FontSize', 20);

    grid on;
    set(gca, 'GridLineStyle', '--', 'GridColor', [0.3, 0.3, 0.3], 'GridAlpha', 0.5, 'LineWidth', 1);

    legend('CFF keypool size', ...
           'CFF keyset size', ...
           'GC keypool size', ...
           'GC keyset size', ...
           'Interpreter', 'latex', 'Location', 'best', ...
             'FontName', 'Open Sans', 'FontSize', 14);

    set(gcf, 'Position', [0, 0, 600, 600]);
    print('numberOfKeysCompare.png', '-dpng', '-r500'); % 保存为 PNG，分辨率为 500 DPI
end


function adjustZAxisForLogScale(runs)
    zlim = 1 / runs;
    h = get(gca, 'Children');
    for i = 1:length(h)
        zdata = get(h(i), 'ZData');
        zdata(zdata < zlim) = zlim;
        set(h(i), 'ZData', zdata);
    end
end


function adjustYAxisForLogScale(runs)
    ylim = 1 / runs;
    h = get(gca, 'Children');
    for i = 1:length(h)
        ydata = get(h(i), 'YData');
        ydata(ydata < ylim) = ylim;
        set(h(i), 'YData', ydata);
    end
end


function [ rgb ] = hex2rgb(hex,range)
    assert(nargin>0&nargin<3,'hex2rgb function must have one or two inputs.')
    if nargin==2
        assert(isscalar(range)==1,'Range must be a scalar, either "1" to scale from 0 to 1 or "256" to scale from 0 to 255.')
    end
    %% Tweak inputs if necessary:
    if iscell(hex)
        assert(isvector(hex)==1,'Unexpected dimensions of input hex values.')

        % In case cell array elements are separated by a comma instead of a
        % semicolon, reshape hex:
        if isrow(hex)
            hex = hex';
        end

        % If input is cell, convert to matrix:
        hex = cell2mat(hex);
    end
    if strcmpi(hex(1,1),'#')
        hex(:,1) = [];
    end
    if nargin == 1
        range = 1;
    end
    %% Convert from hex to rgb:
    switch range
        case 1
            rgb = reshape(sscanf(hex.','%2x'),3,[]).'/255;
        case {255,256}
            rgb = reshape(sscanf(hex.','%2x'),3,[]).';

        otherwise
            error('Range must be either "1" to scale from 0 to 1 or "256" to scale from 0 to 255.')
    end
end

