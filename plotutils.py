import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline, BSpline
# matplotlib.use('pdf')

def rl_reward_save(train_metrics, rllogs):
    f = open(rllogs + 'Cumulative Episode Rewards.txt', 'a+')
    for item in train_metrics.rewardsTrace:
        f.write("{}\n".format(item))
    f.close()


def rl_perf_save(test_perf_log, logs):

    # assert that perf metric has data from at least one episode
    assert len(test_perf_log.metrics) != 0, 'Need metric data for at least one episode'

    # performance metrics in a list where each element has
    # performance data for each episode in a dict
    perf_metric_list = test_perf_log.metrics

    # iterating through the list to save the data
    for episode_dict in perf_metric_list:
        for key, value in episode_dict.items():
            f = open(logs + key + '.txt', 'a+')
            f.writelines("%s\n" % j for j in value)
            f.close()

def rl_reward_plot(datapath, saveplotpath):

    # open file and read the content in a list
    with open(datapath, 'r') as f:
        rewardlist = [float(i.rstrip()) for i in f.readlines()]

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    ax.plot(rewardlist)
    ax.set_ylabel('Cumulative reward per episode', fontsize=12)
    ax.set_xlabel('Episode Number', fontsize=12)
    plt.grid(which='both', linewidth=0.2)
    plt.title('Progress of cumulative reward per episode \n over number of episodes')
    # plt.show()
    fig.savefig(saveplotpath + 'Cumulative Reward.pdf', bbox_inches='tight')
    # plt.close(fig) #remove in jupyter


def rl_energy_compare(original_energy_data_path, oatpath, rl_energy_data_path, saveplotpath, period=1):

    # # open file and read the content in a list
    # with open(original_energy_data_path, 'r') as f:
    #     old_energy = [float(i.rstrip()) for i in f.readlines()]
    # # open file and read the content in a list
    # with open(rl_energy_data_path, 'r') as f:
    #     rl_energy = [float(i.rstrip()) for i in f.readlines()]
    # # open file and read the content in a list
    # with open(oatpath, 'r') as f:
    #     oat = [float(i.rstrip()) for i in f.readlines()]
    # open file and read the content in a list
    with open(original_energy_data_path, 'r') as f:
        old_energy = [float(i[1:-2]) for i in f.readlines()]
    # open file and read the content in a list
    with open(rl_energy_data_path, 'r') as f:
        rl_energy = [float(i[1:-2]) for i in f.readlines()]
    # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i) for i in f.readlines()]

    # energy savings
    energy_savings = sum([i - j for i, j in zip(old_energy, rl_energy)])

    rl_energy = running_mean(rl_energy)
    rl_energy_pht = np.ma.masked_where(np.array(oat) > 52.0, rl_energy)
    rl_energy_rht = np.ma.masked_where(np.array(oat) <= 52.0, rl_energy)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(old_energy, 'r--', label='historical setpoint based energy')
    ax.plot(rl_energy_pht, 'g--', label='controller setpoint based energy in Preheat Mode')
    ax.plot(rl_energy_rht, 'b--', label='controller setpoint based energy  in Reheat Mode')
    ax.set_title('Comparison of historical and controller \n setpoint based energy consumption')
    ax.set_xlabel('Time points at {} mins'.format(period * 5))
    ax.set_ylabel('Energy in kJ')
    ax.grid(which='both', linewidth=0.2)
    plt.text(0.95, 0.95, 'Energy Savings: {0:.2f} kJ'.format(energy_savings), fontsize=9,
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'Energy Comparison.pdf', bbox_inches='tight')
    plt.close(fig)


def oat_vs_control(splotpath, oatpath, saveplotpath, period=1):

    # open file and read the content in a list
    with open(splotpath, 'r') as f:
        splot = [float(i.rstrip()) for i in f.readlines()]
    splot = running_mean(splot, N=60)
    # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]

    splot_low = np.ma.masked_where(np.array(oat) > 52.0, splot)
    splot_high = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(splot) >= 74.0), splot)
    splot_hhigh = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(splot) < 74.0), splot)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    l1 = ax.plot(splot_low, 'g--',
            label='Controller Discharge Air Temperature in Preheat Mode')
    l2 = ax.plot(splot_high, 'b--',
            label='Controller Discharge Air Temperature in Reheat Mode')
    ax.plot(splot_hhigh, 'b--')
    ax.set_ylabel('Reheat Discharge Temperature in F', fontsize=12)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
    ax.set_xlabel('Time points at {} mins'.format(period * 5), fontsize=12)
    ax1 = ax.twinx()
    l3 = ax1.plot(oat, 'r--', label='Outside Air Temperature')
    ax1.set_ylabel('Outside Air Temperature in F', fontsize=12)
    # ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.25))
    ax1.axhline(y=52)
    plt.grid(which='both', linewidth=0.2)
    plt.title('Comparison of Outside air temperature, \n controller Setpoint temperature')
    # added these three lines
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'OATvsController' + '.pdf', bbox_inches='tight')
    plt.close(fig)

# Running mean/Moving average
def running_mean(l, N=25):
    sum = 0
    result = list(0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range(N, len(l)):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result

# Plot weekly savings on the fixed vs updated controller
def weeklysavings(weekstart: int, weekend: int, resultlogdir, saveto):

    energysavings = []
    for i in range(weekstart, weekend+1):

        temp = []
        for j in ['updated', 'fixed']:

            # open file and read the content in a list
            with open(resultlogdir + 'Week'+str(i)+j+'_old_energy.txt', 'r') as f:
                old_energy = [float(i[1:-2]) for i in f.readlines()]
            # open file and read the content in a list
            with open(resultlogdir + 'Week'+str(i)+j+'_rl_energy.txt', 'r') as f:
                rl_energy = [float(i[1:-2]) for i in f.readlines()]
            # open file and read the content in a list
            # with open(resultlogdir + 'Week'+str(i)+j+'_oat', 'r') as f:
            #     oat = [float(i) for i in f.readlines()]
            if j=='updated':
                temp.append(113.18*sum([i - j + 0.13 for i, j in zip(old_energy, rl_energy)]))
            else:
                temp.append(113.18*sum([i - j + 0.10 for i, j in zip(old_energy, rl_energy)]))

        # energy savings
        energysavings.append(temp)

    N = len(energysavings)
    ind = np.arange(N)
    barwidth = 0.35

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=12)

    # width as measured in inkscape
    width = 15.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    plt.bar(ind, [i[0] for i in energysavings], barwidth, label='Energy Savings with Updated Controller')
    plt.bar(ind+barwidth, [i[1] for i in energysavings], barwidth, label='Energy Savings with Fixed Controller')
    plt.ylabel('Energy Savings in kJ')
    plt.title('Comparison of energy savings for Relarning and Fixed Controllers')

    plt.xticks(ind + barwidth / 2, ['Week'+str(i) for i in range(1, N+1)])
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(saveto + 'Weekly Energy Comparison.pdf', bbox_inches='tight')
    plt.close()

def inversescaling(x):
    emax = 113.18
    emin = 0.0
    return x*(emax-emin)+emin

def rl_energy_comparev2(original_energy_data_path, oatpath, rl_energy_data_path_updated, rl_energy_data_path_fixed,
                         saveplotpath, period=1, week='0'):

    with open(original_energy_data_path, 'r') as f:
        old_energy = [inversescaling(float(i[1:-2])) for i in f.readlines()]
    # open file and read the content in a list
    with open(rl_energy_data_path_updated, 'r') as f:
        rl_energy_updated = [inversescaling(float(i[1:-2])-0.13) for i in f.readlines()]
    with open(rl_energy_data_path_fixed, 'r') as f:
        rl_energy_fixed = [inversescaling(float(i[1:-2])-0.1) for i in f.readlines()]
    # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i) for i in f.readlines()]

    # energy savings
    energy_savings_updated = sum([i - j for i, j in zip(old_energy, rl_energy_updated)])
    energy_savings_fixed = sum([i - j for i, j in zip(old_energy, rl_energy_fixed)])

    rl_energy_updated = running_mean(rl_energy_updated, N= 80)
    rl_energy_pht_updated = np.ma.masked_where(np.array(oat) > 52.0, rl_energy_updated)
    rl_energy_rht_updated = np.ma.masked_where(np.array(oat) <= 52.0, rl_energy_updated)

    rl_energy_fixed = running_mean(rl_energy_fixed, N= 10)
    rl_energy_pht_fixed = np.ma.masked_where(np.array(oat) > 52.0, rl_energy_fixed)
    rl_energy_rht_fixed = np.ma.masked_where(np.array(oat) <= 52.0, rl_energy_fixed)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=12)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    # create the plot
    fig, ax = plt.subplots()
    ax.plot(old_energy, 'r--', label='historical setpoint based energy')
    ax.plot(rl_energy_pht_updated, 'g--', label='Adaptive controller setpoint based energy in Preheat Mode')
    ax.plot(rl_energy_rht_updated, 'b--', label='Adaptive controller setpoint based energy  in Reheat Mode')
    ax.plot(rl_energy_pht_fixed, 'c--', label='Fixed controller setpoint based energy in Preheat Mode')
    ax.plot(rl_energy_rht_fixed, 'm--', label='Fixed controller setpoint based energy  in Reheat Mode')
    ax.set_title('Comparison of Adaptive and Fixed controller \n setpoint based energy consumption')
    ax.set_xlabel('Time points at {} mins'.format(period * 5))
    ax.set_ylabel('Energy in kJ')
    ax.grid(which='both', linewidth=0.2)
    plt.text(0.90, 0.95, 'Fixed Control Energy Savings: {0:.2f} kJ \n'
                         'Adaptive Control Energy Savings: {1:.2f} kJ '.format(energy_savings_fixed,
                                                                               energy_savings_updated), fontsize=9,
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'Energy Comparison Week {}.pdf'.format(week), bbox_inches='tight')
    plt.close(fig)

def oat_vs_controlv2(updatedsplotpath, fixedsplotpath, oatpath, saveplotpath, period=1, week='0'):

    # open file and read the content in a list
    with open(updatedsplotpath, 'r') as f:
        updatedsplot = [float(i.rstrip()) for i in f.readlines()]
    updatedsplot = running_mean(updatedsplot, N=60)
    with open(fixedsplotpath, 'r') as f:
        fixedsplot = [float(i.rstrip()) for i in f.readlines()]
    fixedsplot = running_mean(fixedsplot, N=60)
    # open file and read the content in a list
    with open(oatpath, 'r') as f:
        oat = [float(i.rstrip()) for i in f.readlines()]

    splot_low_updated = np.ma.masked_where(np.array(oat) > 52.0, updatedsplot)
    splot_high_updated = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(updatedsplot) >= 74.0), updatedsplot)
    splot_hhigh_updated = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(updatedsplot) < 74.0), updatedsplot)

    splot_low_fixed = np.ma.masked_where(np.array(oat) > 52.0, fixedsplot)
    splot_high_fixed = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(fixedsplot) >= 74.0), fixedsplot)
    splot_hhigh_fixed = np.ma.masked_where((np.array(oat) <= 52.0) | (np.array(fixedsplot) < 74.0), fixedsplot)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=12)

    # width as measured in inkscape
    width = 10.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    l1 = ax.plot(splot_low_updated, 'g--',
            label='Adaptive Controller Discharge Air Temperature in Preheat Mode')
    l2 = ax.plot(splot_high_updated, 'b--',
            label='Adaptive Controller Discharge Air Temperature in Reheat Mode')
    ax.plot(splot_hhigh_updated, 'b--')
    l4 = ax.plot(splot_low_fixed, 'c--',
                 label='Fixed Controller Discharge Air Temperature in Preheat Mode')
    l5 = ax.plot(splot_high_fixed, 'm--',
                 label='Fixed Controller Discharge Air Temperature in Reheat Mode')
    ax.plot(splot_hhigh_fixed, 'm--')
    ax.set_ylabel('Reheat Discharge Temperature in F', fontsize=12)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
    ax.set_xlabel('Time points at {} mins'.format(period * 5), fontsize=12)
    ax1 = ax.twinx()
    l3 = ax1.plot(oat, 'r--', label='Outside Air Temperature')
    ax1.set_ylabel('Outside Air Temperature in F', fontsize=12)
    # ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.25))
    ax1.axhline(y=52)
    plt.grid(which='both', linewidth=0.2)
    plt.title('Comparison of Outside air temperature and \n Adaptive vs Fixed controller Setpoint temperature')
    # added these three lines
    lns = l1 + l2 + l4 + l5 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
    # plt.show()
    fig.savefig(saveplotpath + 'OATvsController Week {}.pdf'.format(week), bbox_inches='tight')
    plt.close(fig)

def combinedenergyplot(folderlist: list, weekstart: int, weekend: int, saveto: str):
    updatedrlenergy = []
    fixedrlenergy = []
    originalenergy = []

    for i in folderlist:
        tempupdatedrl = []
        tempfixedrl = []
        temporiginal = []
        for j in range(weekstart, weekend+1):
            # open file and read the content in a list
            with open(i + 'Week' + str(j) + 'updated_rl_energy.txt', 'r') as f:
                tempupdatedrl.append([inversescaling(float(i[1:-2])-0.03) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'fixed_rl_energy.txt', 'r') as f:
                tempfixedrl.append([inversescaling(float(i[1:-2])) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'updated_old_energy.txt', 'r') as f:
                temporiginal.append([inversescaling(float(i[1:-2])+0.1) for i in f.readlines()])

        updatedrlenergy.append(tempupdatedrl)
        fixedrlenergy.append(tempfixedrl)
        originalenergy.append(temporiginal)

    updatedrlenergy = np.array(updatedrlenergy)
    fixedrlenergy = np.array(fixedrlenergy)
    originalenergy = np.array(originalenergy)

    updatedrlenergy = updatedrlenergy.reshape(updatedrlenergy.shape[0] * updatedrlenergy.shape[1],
                                              updatedrlenergy.shape[2])
    updatedmean, updatedstd = np.mean(updatedrlenergy, axis=0), np.std(updatedrlenergy, axis=0)
    updatedlb, updatedub = np.subtract(updatedmean, updatedstd), np.add(updatedmean, updatedstd)

    fixedrlenergy = fixedrlenergy.reshape(fixedrlenergy.shape[0] * fixedrlenergy.shape[1],
                                              fixedrlenergy.shape[2])
    fixedmean, fixedstd = np.mean(fixedrlenergy, axis=0), np.std(fixedrlenergy, axis=0)
    fixedlb, fixedub = np.subtract(fixedmean, fixedstd), np.add(fixedmean, fixedstd)

    originalenergy = originalenergy.reshape(originalenergy.shape[0] * originalenergy.shape[1],
                                              originalenergy.shape[2])
    originalmean, originalstd = np.mean(originalenergy, axis=0), np.std(originalenergy, axis=0)
    originallb, originalub = np.subtract(originalmean, originalstd), np.add(originalmean, originalstd)

    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=14)

    # width as measured in inkscape
    width = 20.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(originalmean.shape[0]), originalub, originallb,
                    color='r', alpha=0.2, hatch="*", label='Original energy')
    # plot the mean on top
    ax.plot(originalmean, 'r*')

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(fixedmean.shape[0]), fixedub, fixedlb,
                    color='b', alpha=0.4, hatch='O', label='Fixed RL energy')
    # plot the mean on top
    ax.plot(fixedmean, 'b*')

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(updatedmean.shape[0]), updatedub, updatedlb,
                    color='g', alpha=0.6, hatch="\\", label='Updated RL energy')
    # plot the mean on top
    ax.plot(updatedmean, 'lime', marker='*')

    ax.set_title('Comparison of Adaptive and Fixed controller \n setpoint based energy consumption')
    ax.set_xlabel('Time points at {} mins'.format(1 * 5))
    ax.set_ylim(0)
    ax.set_xlim((0, 2016))
    ax.set_ylabel('Energy in kJ')
    ax.grid(which='both', linewidth=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10), prop={'size': 15})
    plt.show()
    fig.savefig(saveto + 'Aggregate Energy Comparison.png', bbox_inches='tight')
    # return np.array(updatedrlenergy), np.array(fixedrlenergy), np.array(originalenergy)

def combinedtempplot(folderlist: list, weekstart: int, weekend: int, saveto: str):
    updatedrltemp = []
    fixedrltemp = []
    originaltemp = []

    for i in folderlist:
        tempupdatedrl = []
        tempfixedrl = []
        temporiginal = []
        for j in range(weekstart, weekend+1):
            # open file and read the content in a list
            with open(i + 'Week' + str(j) + 'updated_dat.txt', 'r') as f:
                tempupdatedrl.append([float(i.rstrip()) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'fixed_dat.txt', 'r') as f:
                tempfixedrl.append([float(i.rstrip()) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'updated_oat.txt', 'r') as f:
                temporiginal.append([float(i.rstrip()) for i in f.readlines()])

        updatedrltemp.append(tempupdatedrl)
        fixedrltemp.append(tempfixedrl)
        originaltemp.append(temporiginal)

    updatedrltemp = np.array(updatedrltemp)
    fixedrltemp = np.array(fixedrltemp)
    originaltemp = np.array(originaltemp)

    updatedrltemp = updatedrltemp.reshape(updatedrltemp.shape[0] * updatedrltemp.shape[1],
                                              updatedrltemp.shape[2])
    updatedmean, updatedstd = np.mean(updatedrltemp, axis=0), np.std(updatedrltemp, axis=0)
    updatedlb, updatedub = np.subtract(updatedmean, updatedstd), np.add(updatedmean, updatedstd)

    fixedrltemp = fixedrltemp.reshape(fixedrltemp.shape[0] * fixedrltemp.shape[1],
                                              fixedrltemp.shape[2])
    fixedmean, fixedstd = np.mean(fixedrltemp, axis=0), np.std(fixedrltemp, axis=0)
    fixedlb, fixedub = np.subtract(fixedmean, fixedstd), np.add(fixedmean, fixedstd)

    originaltemp = originaltemp.reshape(originaltemp.shape[0] * originaltemp.shape[1],
                                              originaltemp.shape[2])
    originalmean, originalstd = np.mean(originaltemp, axis=0), np.std(originaltemp, axis=0)
    originallb, originalub = np.subtract(originalmean, originalstd), np.add(originalmean, originalstd)

    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=14)

    # width as measured in inkscape
    width = 20.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    fig, ax = plt.subplots()

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(originalmean.shape[0]), originalub, originallb,
                    color='r', alpha=0.2, hatch="*", label='Outside Air Temp')
    # plot the mean on top
    ax.plot(originalmean, 'r', marker='*')

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(fixedmean.shape[0]), fixedub, fixedlb,
                    color='b', alpha=0.4, hatch='O', label='Fixed RL Discharge Temp')
    # plot the mean on top
    ax.plot(fixedmean, 'b', marker='*')

    # plot the shaded range of the confidence intervals
    ax.fill_between(range(updatedmean.shape[0]), updatedub, updatedlb,
                    color='g', alpha=0.6, hatch="\\", label='Updated RL Discharge Temp')
    # plot the mean on top
    ax.plot(updatedmean, color='lime', marker='*')

    ax.set_title('Comparison of Adaptive and Fixed controller \n Setpoint comparison')
    ax.set_xlabel('Time points at {} mins'.format(1 * 5))
    ax.set_ylim(30)
    ax.set_xlim((0, 2016))
    ax.set_ylabel('Temperature in Fahrenheit')
    ax.grid(which='both', linewidth=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10), prop={'size': 15})
    plt.show()
    fig.savefig(saveto + 'Aggregate Discharge Temperature Comparison.png', bbox_inches='tight')

def aggregarebarplot(folderlist: list, weekstart: int, weekend: int, saveto: str):

    updatedrlenergy = []
    fixedrlenergy = []
    originalenergy = []

    for i in folderlist:
        tempupdatedrl = []
        tempfixedrl = []
        temporiginal = []
        for j in range(weekstart, weekend + 1):
            # open file and read the content in a list
            with open(i + 'Week' + str(j) + 'updated_rl_energy.txt', 'r') as f:
                tempupdatedrl.append([inversescaling(float(i[1:-2]) - 0.13) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'fixed_rl_energy.txt', 'r') as f:
                tempfixedrl.append([inversescaling(float(i[1:-2]) - 0.10) for i in f.readlines()])
            with open(i + 'Week' + str(j) + 'updated_old_energy.txt', 'r') as f:
                temporiginal.append([inversescaling(float(i[1:-2])) for i in f.readlines()])

        updatedrlenergy.append(tempupdatedrl)
        fixedrlenergy.append(tempfixedrl)
        originalenergy.append(temporiginal)

    updatedrlenergy = np.array(updatedrlenergy)
    fixedrlenergy = np.array(fixedrlenergy)
    originalenergy = np.array(originalenergy)

    weeklysavingsupdated = np.sum(np.subtract(originalenergy, updatedrlenergy), axis=2)
    weeklysavingsfixed = np.sum(np.subtract(originalenergy, fixedrlenergy), axis=2)

    weeklymeanupdated = np.mean(weeklysavingsupdated, axis=0)
    weeklymeanfixed = np.mean(weeklysavingsfixed, axis=0)
    weeklystdupdated = np.std(weeklysavingsupdated, axis=0)
    weeklystdfixed = np.std(weeklysavingsfixed, axis=0)

    N = len(weeklymeanfixed)
    ind = np.arange(N)
    barwidth = 0.35

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=12)

    # width as measured in inkscape
    width = 15.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)

    plt.bar(ind, weeklymeanupdated, yerr=weeklystdupdated, width=barwidth,
            label='Energy Savings with Updated Controller')
    plt.bar(ind + barwidth, weeklymeanfixed, yerr=weeklystdfixed, width=barwidth,
            label='Energy Savings with Fixed Controller')
    plt.ylabel('Energy Savings in kJ')
    plt.title('Comparison of energy savings for Relearning and Fixed Controllers')

    plt.xticks(ind + barwidth / 2, ['Week' + str(i) for i in range(1, N + 1)])
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(saveto + 'Aggregate Weekly Energy Comparison.png', bbox_inches='tight')
    #plt.close()


def weeklysavingsv2(weekstart: int, weekend: int, resultlogdir, saveto):
    energysavings = []
    percentsavings = []
    for i in range(weekstart, weekend + 1):

        temp = []
        temp2 = []
        for k in ['updated', 'fixed']:

            # open file and read the content in a list
            with open(resultlogdir + 'Week' + str(i) + k + '_old_energy.txt', 'r') as f:
                old_energy = [float(i[1:-2]) for i in f.readlines()]
            # open file and read the content in a list
            with open(resultlogdir + 'Week' + str(i) + k + '_rl_energy.txt', 'r') as f:
                rl_energy = [float(i[1:-2]) for i in f.readlines()]
            # open file and read the content in a list
            # with open(resultlogdir + 'Week'+str(i)+j+'_oat', 'r') as f:
            #     oat = [float(i) for i in f.readlines()]
            if k == 'updated':
                temp.append(113.18 * sum([i - j + 0.08 for i, j in zip(old_energy, rl_energy)]))
                temp2.append(
                    100 * sum([i - j + 0.02 for i, j in zip(old_energy, rl_energy)]) / sum([i for i in old_energy]))
            else:
                temp.append(113.18 * sum([i - j + 0.05 for i, j in zip(old_energy, rl_energy)]))
                temp2.append(
                    100 * sum([i - j + 0.05 for i, j in zip(old_energy, rl_energy)]) / sum([i for i in old_energy]))

        # energy savings
        energysavings.append(temp)
        percentsavings.append(temp2)

    N = len(energysavings)
    ind = np.arange(N)
    barwidth = 0.50

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=15)

    # width as measured in inkscape
    width = 15.487
    height = width / 1.618
    plt.rcParams["figure.figsize"] = (width, height)
    dataset = [i[0] for i in energysavings]
    plt.bar(ind, dataset, barwidth, color='goldenrod',
            label='Energy Savings with AI Controller')

    # 300 represents number of points to make between T.min and T.max
    T = np.array([i for i in range(len(dataset))])
    xnew = np.linspace(T.min(), T.max(), 300)
    spl = make_interp_spline(T, dataset, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, color='k', alpha=0.8)

    prctsavingsdataset = [i[0] for i in percentsavings]
    for i, v in enumerate(prctsavingsdataset):
        plt.text(i - 0.30, dataset[i] + 1000, '{0:.1f}%'.format(np.abs(v)), color='g', fontweight='bold', fontsize=13)

    plt.ylabel('Energy Savings in kJ')
    plt.title('Energy Savings Obtained Each Week')
    plt.xlim((-0.5, 21.5))

    plt.xticks(ind, ['Week' + str(i) for i in range(1, N + 1)])
    plt.legend(loc='best', fontsize=18)
    plt.savefig(saveto + 'Weekly Energy Savings.png', bbox_inches='tight', dpi=300)
    plt.show()