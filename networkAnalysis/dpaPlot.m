set(groot, 'defaultFigureCloseRequestFcn', 'close(gcf)');   % avoid R2023b crash after closing figures


%plot_3d_for_dpa_fix();
%plot_3d_for_dpa_random();
%plot_3d_for_tpa_random();

plot_for_dpa_our();


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


function plot_for_dpa_our()
    m_runs = 10^6;
    m_max_compromised_nodes = 8;
    m_max_keyset_size = 11;
    m_num_nodes = 10;
    m_folder_cff = '/Users/xingyuzhou/Downloads/dpa_cff_our_106_22/merged';
    m_folder_mhd = '/Users/xingyuzhou/Downloads/dpa_mhd_our_106/merged';

    success_ratio_vector_cff = zeros(1, m_max_compromised_nodes);
    data_count_checks = cell(1, m_max_compromised_nodes);

%    success_ratio_matrix_mhd = zeros(m_max_keyset_size, m_max_compromised_nodes);
%    avg_checks_matrix_mhd = zeros(m_max_keyset_size, m_max_compromised_nodes);

    for compromised_nodes = 1:1:m_max_compromised_nodes
        filename = sprintf('%s/csv_dpa_cff_our_%druns_%dc.csv', m_folder_cff, m_runs, compromised_nodes);
        data = readtable(filename);

        success_ratio_vector_cff(compromised_nodes) = sum(data.is_success == 1) / m_runs;
        data_count_checks{compromised_nodes} = data.count_checks;

%        for keyset_size = 1:2:m_max_keyset_size
%            filename = sprintf('%s/csv_dpa_mhd_our_%druns_%dc_%dkeys.csv', m_folder_mhd, m_runs, compromised_nodes, keyset_size);
%            data = readtable(filename);
%            success_ratio_matrix_mhd(keyset_size, compromised_nodes) = sum(data.is_success == 1) / m_runs;
%            avg_checks_matrix_mhd(keyset_size, compromised_nodes) = sum(data.count_checks) / m_runs;
%        end

    end

    fig1 = figure(1);
    subplot(1, 2, 1);
    bar(success_ratio_vector_cff);
    set(gca(), 'YScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Success Ratio');
    title(sprintf('CFF, Our DPA Model\n%d runs / compromised nodes (Total Nodes = %d)', m_runs, m_num_nodes));
    adjustYAxisForLogScale(m_runs)    % 调整 Y 轴数据以适应对数刻度

    subplot(1, 2, 2);
    boxplot(cell2mat(data_count_checks), 'Labels', 1:m_max_compromised_nodes);
    xlabel('Compromised Nodes');
    ylabel('Checks Count');
    title(sprintf('CFF, Our DPA Model\n%d runs / compromised nodes (Total Nodes = %d)', m_runs, m_num_nodes));

    fig1.Position = [0, 0, 1200, 600];
    print(fig1, '-dpng', sprintf('%s/plot_cff_%druns.png', m_folder_cff, m_runs), '-r300');

%    fig2 = figure(2);
%    subplot(2, 2, 3);
%    bar3(success_ratio_matrix_mhd);
%    set(gca(), 'ZScale', 'log');
%    xlabel('Compromised Nodes');
%    ylabel('Relay Node Keyset Size');
%    zlabel('Success Ratio');
%    title(sprintf('MHD, Our DPA Model\n%d runs / keyset size / compromised nodes (Total Nodes = %d)', m_runs, m_num_nodes));
%    view(58, 16); % azimuth, elevation
%    adjustZAxisForLogScale(m_runs)    % 调整 Z 轴数据以适应对数刻度
%
%    subplot(2, 2, 4);
%    bar3(avg_checks_matrix_mhd);
%    xlabel('Compromised Nodes');
%    ylabel('Relay Node Keyset Size');
%    zlabel('Average Checks');
%    title(sprintf('MHD, Our DPA Model\n%d runs / keyset size / compromised nodes (Total Nodes = %d)', m_runs, m_num_nodes));
%    view(58, 16); % azimuth, elevation
%
%    fig2.Position = [0, 0, 1200, 1200];
%    print(fig2, '-dpng', sprintf('%s/plot_mhd_%druns.png', m_folder_mhd, m_runs), '-r300');
end


function plot_3d_for_dpa_random()
    m_runs = 10000;
    m_key_pool_size = 12;
    m_max_compromised_nodes = 10;
    m_num_nodes = 12;
    m_folder = '/Users/xingyuzhou/Downloads/count1w/merged';

    max_subset_size = m_key_pool_size - 1;

    success_ratio_matrix1 = zeros(max_subset_size, m_max_compromised_nodes);
    avg_checks_vetor1 = zeros(1, m_max_compromised_nodes);

%    success_ratio_matrix2 = zeros(max_subset_size, max_compromised_nodes);
%    success_ratio_matrix3 = zeros(max_subset_size, max_compromised_nodes);

    for compromised_nodes = 1:m_max_compromised_nodes
        checks = 0;
        for keys = 1:max_subset_size
            filename1 = sprintf('%s/csv_dpa_mhd_n_our_%druns_%dc_%dkeys.csv', m_folder, m_runs, compromised_nodes, keys);
            data1 = readtable(filename1);
            success_count1 = sum(data1.is_success == 1);
            success_ratios1 = success_count1 / m_runs;
            success_ratio_matrix1(keys, compromised_nodes) = success_ratios1;
            checks = checks + sum(data1.count_checks);

%            filename2 = sprintf('./%s/csv_mhd_n_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
%            data2 = readtable(filename2);
%            success_count2 = sum(data2.is_success == 1);
%            success_ratios2 = success_count2 / runs;
%            success_ratio_matrix2(keys, compromised_nodes) = success_ratios2;


%            filename3 = sprintf('./%s/csv_rd_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
%            data3 = readtable(filename3);
%            success_count3 = sum(data3.is_success == 1);
%            success_ratios3 = success_count3 / runs;
%            success_ratio_matrix3(keys, compromised_nodes) = success_ratios3;
        end
        avg_checks_vetor1(compromised_nodes) = checks / (max_subset_size * m_runs);
    end

    subplot(2, 1, 1);
    bar3(success_ratio_matrix1);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Our DPA Model) / keyset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));
    view(58, 16); % azimuth, elevation
    adjustZAxisForLogScale(m_runs)    % 调整 Z 轴数据以适应对数刻度

    subplot(2, 1, 2);
    plot(avg_checks_vetor1);
    xlabel('Compromised Nodes');
    ylabel('Average Checks');
    title(sprintf('%d runs (Max-Hamming Dist. Our DPA Model) / keyset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));

%    subplot(3, 1, 2);
%    bar3(success_ratio_matrix2);
%    set(gca(), 'ZScale', 'log');
%    xlabel('Compromised Nodes');
%    ylabel('Relay Node Keyset Size');
%    zlabel('Success Ratio');
%    title(sprintf('%d runs (Max-Hamming Dist. Other DPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
%    view(83, 22); % azimuth, elevation
%    adjustZAxisForLogScale(runs)

%    subplot(3, 1, 3);
%    bar3(success_ratio_matrix3);
%    set(gca(), 'ZScale', 'log');
%    xlabel('Compromised Nodes');
%    ylabel('Relay Node Keyset Size');
%    zlabel('Success Ratio');
%    title(sprintf('%d runs (Random Dist. Other DPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
%    view(-124, 16); % azimuth, elevation
%    adjustZAxisForLogScale(runs)

    % 设置图形尺寸并保存
    fig = gcf;
%    fig.Visible = 'off';
    fig.Position = [0, 0, 800, 1600];
    print(fig, '-dpng', sprintf('%s/plot_3d_dpa_%druns.png', m_folder, m_runs), '-r300');

end

function plot_3d_for_tpa_random()
    m_runs = 100000;
    m_key_pool_size = 12;
    m_max_compromised_nodes = 3;
    m_num_nodes = 12;
    m_folder = '/Users/xingyuzhou/TUNC/networkAnalysis/tpa10w';

    max_subset_size = m_key_pool_size - 1;
    success_ratio_matrix1 = zeros(max_subset_size, m_max_compromised_nodes);
    success_ratio_matrix2 = zeros(max_subset_size, m_max_compromised_nodes);
    success_ratio_matrix3 = zeros(max_subset_size, m_max_compromised_nodes);
    success_ratio_matrix4 = zeros(max_subset_size, m_max_compromised_nodes);

    for compromised_nodes = 1:m_max_compromised_nodes
        for keys = 1:max_subset_size
            filename1 = sprintf('%s/csv_tpa_mhd_n_our_%druns_%dc_%dkeys.csv', m_folder, m_runs, compromised_nodes, keys);
            data1 = readtable(filename1);
            success_count1 = sum(data1.is_success == 1);
            success_ratios1 = success_count1 / m_runs;
            success_ratio_matrix1(keys, compromised_nodes) = success_ratios1;


            filename2 = sprintf('%s/csv_tpa_mhd_n_other_%druns_%dc_%dkeys.csv', m_folder, m_runs, compromised_nodes, keys);
            data2 = readtable(filename2);
            success_count2 = sum(data2.is_success == 1);
            success_ratios2 = success_count2 / m_runs;
            success_ratio_matrix2(keys, compromised_nodes) = success_ratios2;


            filename3 = sprintf('%s/csv_tpa_rd_our_%druns_%dc_%dkeys.csv', m_folder, m_runs, compromised_nodes, keys);
            data3 = readtable(filename3);
            success_count3 = sum(data3.is_success == 1);
            success_ratios3 = success_count3 / m_runs;
            success_ratio_matrix3(keys, compromised_nodes) = success_ratios3;

            filename4 = sprintf('%s/csv_tpa_rd_other_%druns_%dc_%dkeys.csv', m_folder, m_runs, compromised_nodes, keys);
            data4 = readtable(filename4);
            success_count4 = sum(data4.is_success == 1);
            success_ratios4 = success_count4 / m_runs;
            success_ratio_matrix4(keys, compromised_nodes) = success_ratios4;
        end
    end

    subplot(4, 1, 1);
    bar3(success_ratio_matrix1);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Our TPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));
    view(69, 12); % azimuth, elevation
    adjustZAxisForLogScale(m_runs)    % 调整 Z 轴数据以适应对数刻度

    subplot(4, 1, 2);
    bar3(success_ratio_matrix2);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Other TPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));
    view(-57, 9); % azimuth, elevation
    adjustZAxisForLogScale(m_runs)

    subplot(4, 1, 3);
    bar3(success_ratio_matrix3);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Random Dist. Our TPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));
    view(64, 21); % azimuth, elevation
    adjustZAxisForLogScale(m_runs)

    subplot(4, 1, 4);
    bar3(success_ratio_matrix4);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Relay Node Keyset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Random Dist. Other TPA Model) / Relay Node Keyset Size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', m_runs, m_num_nodes, m_key_pool_size));
    view(-74, 18); % azimuth, elevation
    adjustZAxisForLogScale(m_runs)

    % 设置图形尺寸并保存
    fig = gcf;
%    fig.Visible = 'off';
    fig.Position = [0, 0, 700, 1400];
    print(fig, '-dpng', sprintf('%s/plot_3d_tpa_%druns.png', m_folder, m_runs), '-r300');

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

