set(groot, 'defaultFigureCloseRequestFcn', 'close(gcf)');   % avoid R2023b crash after closing figures


%plot_3d_for_dpa_fix();
plot_3d_for_dpa_random();

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

    % 设置图形尺寸并保存
    fig = gcf;
    fig.Position = [0, 0, 800, 600];
    print(fig, '-dpng', sprintf('./dpa_fix/plot_3d_%druns.png', runs), '-r300');
end


function plot_3d_for_dpa_random()
    runs = 100000;
    max_subset_size = 8;
    max_compromised_nodes = 3;

    total_nodes = 12;

    key_pool_size = max_subset_size + 1;

    success_ratio_matrix1 = zeros(max_subset_size, max_compromised_nodes);
    success_ratio_matrix2 = zeros(max_subset_size, max_compromised_nodes);

    for compromised_nodes = 1:max_compromised_nodes
        for keys = 1:max_subset_size
            filename1 = sprintf('./results/dpa_random_pos/csv_mhd_%druns_%dc_%dkeys.csv', runs, compromised_nodes, keys);
            filename2 = sprintf('./results/dpa_random_pos/csv_rd_%druns_%dc_%dkeys.csv', runs, compromised_nodes, keys);
            data1 = readtable(filename1);
            data2 = readtable(filename2);
            success_count1 = sum(data1.is_success == 1);
            success_count2 = sum(data2.is_success == 1);
            success_ratios1 = success_count1 / runs;
            success_ratios2 = success_count2 / runs;

            success_ratio_matrix1(keys, compromised_nodes) = success_ratios1;
            success_ratio_matrix2(keys, compromised_nodes) = success_ratios2;
        end
    end

    subplot(1, 2, 1);
    bar3(success_ratio_matrix1);

    ax1 = gca;
    set(ax1, 'ZScale', 'log');

    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Max-Hamming Dist.) / key subset size / compromised nodes\n(Total Nodes = 12, Key Pool Size = 8)', runs));
    view(-124, 16); % azimuth, elevation

    % 调整 Z 轴数据以适应对数刻度
    adjustZAxisForLogScale(runs)

    subplot(1, 2, 2);
    bar3(success_ratio_matrix2);

    ax2 = gca;
    set(ax2, 'ZScale', 'log');

    xlabel('Compromised Nodes');
    ylabel('Key Subset Size');
    zlabel('Success Ratio');
    title(sprintf('%d runs (Random Dist.) / key subset size / compromised nodes\n(Total Nodes = 12, Key Pool Size = 8)', runs));
    view(-124, 16); % azimuth, elevation

    % 调整 Z 轴数据以适应对数刻度
    adjustZAxisForLogScale(runs)

    % 设置图形尺寸并保存
    fig = gcf;
    fig.Position = [0, 0, 1200, 800];
    print(fig, '-dpng', sprintf('./results/dpa_random_pos/plot_3d_%druns.png', runs), '-r500');

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

