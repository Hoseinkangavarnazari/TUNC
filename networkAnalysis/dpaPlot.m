set(groot, 'defaultFigureCloseRequestFcn', 'close(gcf)');   % avoid R2023b crash after closing figures


%plot_3d_for_dpa_fix();
plot_3d_for_dpa_random();

function plot_3d_for_dpa_fix()
    runs = 100000;
    max_subset_size = 9;
    hops = 8;

    success_ratio_matrix = zeros(max_subset_size, hops);

    for hop = 1:hops
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

    xlabel('Hops');
    ylabel('Key Subset Size');
    zlabel('Logarithmic Success Ratio');
    title(sprintf('%d runs / key subset size / hop', runs));
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
    max_subset_size = 9;
    max_compromised_nodes = 5;
    key_pool_size = max_subset_size + 1;

    success_ratio_matrix = zeros(max_subset_size, max_compromised_nodes);

    for compromised_nodes = 1:max_compromised_nodes
        for keys = 1:max_subset_size
            filename = sprintf('./dpa_random/results_%dc_%druns_%dkeys.csv', compromised_nodes, runs, keys);
            data = readtable(filename);
            success_count = sum(data.is_success == 1);
            success_ratios = success_count / runs;

            success_ratio_matrix(keys, compromised_nodes) = success_ratios;
        end
    end

    bar3(success_ratio_matrix);

    ax = gca;
    set(ax, 'ZScale', 'log');

    xlabel('Compromised Nodes');
    ylabel(sprintf('Key Subset Size (Key Pool Size = %d)', key_pool_size));
    zlabel('Logarithmic Success Ratio');
    title(sprintf('%d runs / key subset size / compromised nodes', runs));
    view(-124, 16); % azimuth, elevation

    % 调整 Z 轴数据以适应对数刻度
    zlim = 10^(-4);
    h = get(gca, 'Children');
    for i = 1:length(h)
        zdata = get(h(i), 'ZData');
        zdata(zdata < zlim) = zlim;
        set(h(i), 'ZData', zdata);
    end

    % 设置图形尺寸并保存
    fig = gcf;
    fig.Position = [0, 0, 800, 600];
    print(fig, '-dpng', sprintf('./dpa_random/plot_3d_%druns.png', runs), '-r300');

end

