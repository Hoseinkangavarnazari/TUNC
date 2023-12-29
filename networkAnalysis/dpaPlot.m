set(groot, 'defaultFigureCloseRequestFcn', 'close(gcf)');   % avoid R2023b crash after closing figures


%plot_3d_for_dpa_fix();
%plot_3d_for_dpa_random();
plot_3d_for_tpa_random();


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
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs / key subset size / hop\n(Max Hops = %d, Key Pool Size = %d)', runs, max_hop, max_subset_size + 1));
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


function plot_3d_for_dpa_random()
    runs = 1000;
    key_pool_size = 10;
    max_compromised_nodes = 3;
    num_nodes = 10;
    folder = 'temp2';

    max_subset_size = key_pool_size - 1;

    success_ratio_matrix1 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix2 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix3 = zeros(max_subset_size, max_compromised_nodes);

    for compromised_nodes = 1:max_compromised_nodes
        for keys = 1:max_subset_size
            filename1 = sprintf('./%s/csv_mhd_n_our_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data1 = readtable(filename1);
            success_count1 = sum(data1.is_success == 1);
            success_ratios1 = success_count1 / runs;
            success_ratio_matrix1(keys, compromised_nodes) = success_ratios1;


            filename2 = sprintf('./%s/csv_mhd_n_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data2 = readtable(filename2);
            success_count2 = sum(data2.is_success == 1);
            success_ratios2 = success_count2 / runs;
            success_ratio_matrix2(keys, compromised_nodes) = success_ratios2;


%            filename3 = sprintf('./%s/csv_rd_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
%            data3 = readtable(filename3);
%            success_count3 = sum(data3.is_success == 1);
%            success_ratios3 = success_count3 / runs;
%            success_ratio_matrix3(keys, compromised_nodes) = success_ratios3;
        end
    end

    subplot(3, 1, 1);
    bar3(success_ratio_matrix1);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Our DPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(83, 22); % azimuth, elevation
    adjustZAxisForLogScale(runs)    % 调整 Z 轴数据以适应对数刻度

    subplot(3, 1, 2);
    bar3(success_ratio_matrix2);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Other DPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(83, 22); % azimuth, elevation
    adjustZAxisForLogScale(runs)

%    subplot(3, 1, 3);
%    bar3(success_ratio_matrix3);
%    set(gca(), 'ZScale', 'log');
%    xlabel('Compromised Nodes');
%    ylabel('Key Subset Size');
%    zlabel('Success Ratio');
%    title(sprintf('%d runs (Random Dist. Other DPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
%    view(-124, 16); % azimuth, elevation
%    adjustZAxisForLogScale(runs)

    % 设置图形尺寸并保存
    fig = gcf;
%    fig.Visible = 'off';
    fig.Position = [0, 0, 500, 1000];
    print(fig, '-dpng', sprintf('./%s/plot_3d_dpa_%druns.png', folder, runs), '-r300');

end

function plot_3d_for_tpa_random()
    runs = 100000;
    key_pool_size = 12;
    max_compromised_nodes = 3;
    num_nodes = 12;
    folder = '/Users/xingyuzhou/TUNC/networkAnalysis/tpa10w';

    max_subset_size = key_pool_size - 1;

    success_ratio_matrix1 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix2 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix3 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix4 = zeros(max_subset_size, max_compromised_nodes);

    for compromised_nodes = 1:max_compromised_nodes
        for keys = 1:max_subset_size
            filename1 = sprintf('%s/csv_tpa_mhd_n_our_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data1 = readtable(filename1);
            success_count1 = sum(data1.is_success == 1);
            success_ratios1 = success_count1 / runs;
            success_ratio_matrix1(keys, compromised_nodes) = success_ratios1;


            filename2 = sprintf('/%s/csv_tpa_mhd_n_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data2 = readtable(filename2);
            success_count2 = sum(data2.is_success == 1);
            success_ratios2 = success_count2 / runs;
            success_ratio_matrix2(keys, compromised_nodes) = success_ratios2;


            filename3 = sprintf('/%s/csv_tpa_rd_our_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data3 = readtable(filename3);
            success_count3 = sum(data3.is_success == 1);
            success_ratios3 = success_count3 / runs;
            success_ratio_matrix3(keys, compromised_nodes) = success_ratios3;

            filename4 = sprintf('/%s/csv_tpa_rd_other_%druns_%dc_%dkeys.csv', folder, runs, compromised_nodes, keys);
            data4 = readtable(filename4);
            success_count4 = sum(data4.is_success == 1);
            success_ratios4 = success_count4 / runs;
            success_ratio_matrix4(keys, compromised_nodes) = success_ratios4;
        end
    end

    subplot(4, 1, 1);
    bar3(success_ratio_matrix1);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Our TPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(69, 12); % azimuth, elevation
    adjustZAxisForLogScale(runs)    % 调整 Z 轴数据以适应对数刻度

    subplot(4, 1, 2);
    bar3(success_ratio_matrix2);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist. Other TPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(-57, 9); % azimuth, elevation
    adjustZAxisForLogScale(runs)

    subplot(4, 1, 3);
    bar3(success_ratio_matrix3);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Random Dist. Our TPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(64, 21); % azimuth, elevation
    adjustZAxisForLogScale(runs)

    subplot(4, 1, 4);
    bar3(success_ratio_matrix4);
    set(gca(), 'ZScale', 'log');
    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Random Dist. Other TPA Model) / key subset size / compromised nodes\n(Total Nodes = %d, Key Pool Size = %d)', runs, num_nodes, key_pool_size));
    view(-74, 18); % azimuth, elevation
    adjustZAxisForLogScale(runs)

    % 设置图形尺寸并保存
    fig = gcf;
%    fig.Visible = 'off';
    fig.Position = [0, 0, 700, 1400];
    print(fig, '-dpng', sprintf('/%s/plot_3d_tpa_%druns.png', folder, runs), '-r300');

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

