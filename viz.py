import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from datetime import date
from itertools import product

from data_keeper import data_keeper
from met_monitor import met_monitor
import met_fitting as mfit


class met_viz:

    def __init__(self, tower, masks=True):
        # Create a connection to the server and if the tables don't exist create them 
        self.dk = data_keeper(verbose=False)
        self.met_df = self.dk.get_timeseries(tower)
        self.size = self.met_df.shape[0] # size of retrieved df

        self.mm = met_monitor()
        
        print('\tShape of retrieved dataframe: (%i, %i).' %self.met_df.shape)
        # if masks is True:
        #     self.met_masks = self.dk.get_masks(tower)
        #     print('Shape of retrieved msk      : (%i, %i). \n' %self.met_masks.shape)

    # combine two or more masks
    def combine_masks(self, *masks):
        mask = [ ma.any(m) for m in zip(*masks) ]
        return mask


    def print_mask_stats(self, d, n):
        # d is a dict w/ 'flagname': count
        # n is the total number of data points

        # update d to be percent
        d.update((k, v * (100. / n)) for k,v in d.items())
        
        if d['outliers'] != 0: # print anomaly value if that has been set
            print('\tFlagged  : %10s %10s %10.2f%%  %8i' \
                  %('--', '--', d['outliers'], n))
        else:
            keys = [ k for k in d.keys() ]
            print('\tmaskNames: {:12s} {:12s} {:12s} Total'.format(*keys))
            print('\tFlagged  : %10.2f%% %10.2f%% %10s %8i'\
                  %(d['sensorBounds'], d['flatlining'], '--', n))
        return

    # define a `scatter` plot method for XYZ 
    def scatter(self, x, y, z, label=None, ax=None, zrange=None,
                   s=10, marker='+', alpha=0.5, cmap='jet'):

        if zrange is None: zrange = (-30, 40) # default temperature range
        
        im = ax.scatter(x, y, marker=marker, alpha=alpha, label=label)
        #                        cmap=cmap, vmin=zrange[0], vmax=zrange[1])
        return im
    
    def plot_masked(self, x, y, t, y1, y2, label=None, axes=None):
        # dict that assigns a color to each flag
        d = {'sensorBounds': 'g',
             'flatlining': 'b',
             'minSpeed': 'r',
             'outliers': 'gray',
             'None': 'black'}

        # plot them -- use zip for abstraction
        # wdir vs. wspd
        axes[0,0].scatter(x, y,  c=d[label], label=label, alpha=1, s=10, marker='+')
        # wdir timeseries
        axes[1,1].scatter(t, x,  c=d[label], label=label, alpha=1, s=10, marker='+')
        # wspd1 timeseries
        axes[2,1].scatter(t, y1, c=d[label], label=label, alpha=1, s=10, marker='+')
        # wspd1 timeseries
        axes[3,1].scatter(t, y2, c=d[label], label=label, alpha=1, s=10, marker='+')

        return


    def get_mask(self, maskName, x, y1, y2):

        # define nullMask when masking is not applicable -- get rid when zip is implemented?
        nullMask = ma.zeros(len(x), dtype=bool)
        
        # use zip here -- be cognizant of what field
        if maskName == 'sensorBounds':
            m1 = self.mm.mask_sensorBounds(x,  'wdir_avg')
            m2 = self.mm.mask_sensorBounds(y1, 'wspd_avg')
            m3 = self.mm.mask_sensorBounds(y2, 'wspd_avg')
        elif maskName == 'flatlining':
            m1 = self.mm.mask_flatlining(x)
            m2 = self.mm.mask_flatlining(y1)
            m3 = self.mm.mask_flatlining(y2)
        elif maskName == 'outliers': # for now
            return nullMask
        else:
            print('I have not been taught how to process `%s` mask.' %maskName)
            return nullMask
            
        # once connected to MySQL DB, this mask would be stored and retrieved later when needed
        m0 = self.combine_masks(m1, m2, m3)

        return m0
        
    # return a full mask, plot/print each mask if specified
    # this routine needs a layer of abstraction as each mask does the same operation
    def process_masks(self, x, y1, y2, t, axes=None,
                      masks=None, plot=False, print=False):

        # calc y from y1, y2
        y = y1 / y2

        # total mask
        fullmask = ma.zeros(len(x), dtype=bool)
        # dict to count how many are flagged
        d_masked = {'sensorBounds': 0,
                    'flatlining': 0,
                    'outliers': 0}

        for maskName in masks:
            m0 = self.get_mask(maskName, x, y1, y2)
            xm = ma.array(x, mask=m0)
            d_masked[maskName] = ma.count_masked(xm)
            
            fullmask = self.combine_masks(fullmask, m0)
            
            if (plot is True) & (x.data[xm.mask == True].shape[0] > 0):
                self.plot_masked(x.data[xm.mask == True],  y.data[xm.mask == True], t.data[xm.mask == True],
                                 y1.data[xm.mask == True], y2.data[xm.mask == True],
                                 axes=axes, label=maskName)
                            
        if print:
            self.print_mask_stats(d_masked, len(x))

        return (fullmask, d_masked)


    # plot the time series
    def plot_ts(self, x, y, z, mask, yname, ax=None, plot_masked=True, time_range=None):

        ax.set_xlim(time_range[0], time_range[1])
        zrange = (-30, 40) # range of temperature for colorbar
        
        if 'wspd' in yname:
            sensorName = 'wspd_avg'
            yrange = (-1, 30)
            ax.set_ylim(yrange)
            ax.set_yticks(np.arange(yrange[0], yrange[1]+1, 10))
            ax.set_ylabel('Average Wind Speed / 10 min (m/s)')
        if 'wdir' in yname:
            sensorName = 'wdir_avg'
            yrange = (0, 360)
            ax.set_ylim(yrange)
            ax.set_yticks(np.arange(yrange[0], yrange[1]+1, 90))
            ax.set_ylabel('Average Wind Direction / 10 min (deg)')

        # if plot_masked is True:
        #     self.plot_masked(x, y, label='flatlining', ax=ax)

        # plot the data
        im = self.scatter(x, y, z, ax=ax, marker='+')

        return

    
    def dashboard(self, xname, yname1, yname2, zname=False, time_range=False,
                  fit=False, unpickle=False, 
                  masks=False, plot=False, print=False,
                  plot_anomaly=False, set_outlier=False, **kwargs):
        
        fig, axes = plt.subplots(4, 3, figsize=(20, 12),
                                 gridspec_kw = {'width_ratios':[2, 6, 1],
                                                'height_ratios':[1.5, 1, 1, 1]})
        fig.tight_layout()
        nullfmt = NullFormatter()         # no labels

        df = self.met_df.copy()
        
        if time_range is not False:
            df = df[time_range[0]: time_range[1]]

        # create masked versions of the vars
        x = ma.masked_invalid(df[xname])
        y1, y2 =  ma.masked_invalid(df[yname1]), ma.masked_invalid(df[yname2])
        y = y1 / y2
        if zname is False: zname = 'temp_avg_1a'
        z = ma.masked_invalid(df[zname])
        t = ma.array(df.index)

        # get, plot, & print mask for all specified masks
        # d_masked is a dict with a count of no. of masked points
        mask, d_masked = self.process_masks(x, y1, y2, t, axes=axes,
                                            masks=('sensorBounds', 'flatlining'),
                                            plot=plot, print=print)                   
        # # reconstruct x, y, z, t with the returned mask
        # x, y, z, t = ma.array(x, mask=mask), ma.array(y, mask=mask), \
        #              ma.array(z, mask=mask), ma.array(t, mask=mask)

        ### PLOT 1 -- winddirection vs. windspeed ratio
        ax = axes[0, 0]
        ax.set_xlabel('Wind Direction (deg)')
        ax.set_xlim(0, 360)
        ax.xaxis.set_ticks(np.arange(0, 361, 45))
        ax.set_ylabel('Windspeed Ratio')
        ax.set_ylim(0.5, 1.5)
        zrange = [-30, 40]

        # mask out small speeds for fitting purposes
        mask_minSpeed = self.combine_masks(self.mm.mask_minSpeed(y1),
                                           self.mm.mask_minSpeed(y2))
        mask1 = self.combine_masks(mask, mask_minSpeed)
        
        # fit the data -- scikit learn ignores masks, so beware
        if fit in ('knn', 'k'):
            # check if n_neighbors, cv are in kwargs
            nn = kwargs['n_neighbors'] if 'n_neighbors' in kwargs else None                
            cv = kwargs['cv'] if 'cv' in kwargs else None
            scoring = kwargs['scoring'] if 'scoring' in kwargs else None          
            
            # do the fit
            x_fit, y_fit, sigma = mfit.fit_knn(ma.compressed(ma.array(x, mask=mask1)),
                                               ma.compressed(ma.array(y, mask=mask1)),
                                               axes=axes,
                                               n_neighbors=nn, cv=cv, scoring=scoring,
                                               unpickle=unpickle)
            
        elif fit in ('lowess', 'l'):
            x_fit, y_fit = mfit.fit_lowess(x, y)
        elif fit in ('savgol', 's'):
            x_fit, y_fit = x, mfit.fit_savgol(y)

        # sort the returned arrays for plotting purposes
        indx = x_fit.argsort()
        # confidence interval
        CI = 1.9600 * sigma[indx] # 2-sigma CI
        ax.fill_between(x_fit[indx], y_fit[indx] + CI, y_fit[indx] - CI,
                        alpha=0.3, color='darkorange', edgecolor='', label='95% CI')
        # plot the fit                
        ax.plot(x_fit[indx], y_fit[indx], alpha=1, color='red', label=fit)
        ax.legend(bbox_to_anchor=(0.2, -0.5), loc=3, borderaxespad=0.)

        # plot the mask_minSpeed data
        ym = ma.array(y, mask=mask_minSpeed)
        ax.scatter(x[ym.mask == True], y[ym.mask == True],
                   color='black', marker='o', s=10, alpha=0.3)
        # plot the data
        ax.scatter(ma.array(x, mask=mask1), ma.array(y, mask=mask1),
                        color='b', marker='+', s=10, alpha=0.5)

        # plot the color bar
        # zname = 'Temperature (C)'
        # cbar_ax = fig.add_axes([0.02, 0.05, 0.02, 0.60])
        # cbar_ticks = np.arange(zrange[0], zrange[1]+10, 10)

        # cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
        # cbar.set_label('%s' %zname, rotation=270)
        #self.plot_cbar(fig, im, zrange)
        #fig.subplots_adjust(right=0.92)
        
        ### PLOT 2
        ax = axes[0, 1]
        # now, plot the normalized deviation -- anomaly
        ax.set_ylabel('Normalized Deviation')
        yrange = (-5, 5)
        ax.set_ylim(yrange)
        ax.set_xlim(time_range[0], time_range[1])
        ax.set_yticks(np.arange(yrange[0], yrange[1]+0.1, 1))
        
        # calculate the normalized deviation -- anomaly
        norm_dev = ma.array( ((ma.compressed(ma.array(y, mask=mask1)) - y_fit ) / sigma))

        #plot
        ax.scatter(ma.compressed(ma.array(t, mask=mask1)), norm_dev, 
                   marker='+', s=10, alpha=0.5)
        
        if 'outliers':
            label = 'mask_outliers'

            if set_outlier is True:
                maxPercentile = 99
                maxDev = np.percentile(np.abs(norm_dev), maxPercentile)
                print('\t\tThe 3-sigma (>%2i%%) norm deviation is: %5.2f. Set for production.'
                      %(maxPercentile, maxDev))

                ## pickle maxDev??
            else:
                # we need to decide which of the three maxDev to use.
                # The easiest would be to pickle all three and use a different maxDev for each height
                # for now, let this be the palce holder
                maxDev = 4.28
                
            mask4 = self.mm.mask_outliers(norm_dev, maxDev=maxDev)

            if print is True:
                d_masked['outliers'] = ma.count_masked(ma.array(norm_dev, mask=mask4))
                self.print_mask_stats(d_masked, len(norm_dev))
                
        ### PLOT 3
        # plot the histogram on the side
        ax = axes[0, 2]
        ax.set_ylim(yrange)
        ax.set_yticks(np.arange(yrange[0], yrange[1]+0.1, 1))
        ax.set_xlim((1, 10000))
        
        # now determine nice limits by hand:
        binwidth = 0.2
        bins = np.arange(yrange[0], yrange[1], binwidth)
        
        # only plot +/-5 for hist
        ax.hist(norm_dev[np.abs(norm_dev) < 5], bins=bins, orientation='horizontal')
        ax.set_xscale('log')

        ### PLOT 4-6 -- time series
        self.plot_ts(t, x,  z, mask, 'wdir_avg', ax=axes[1, 1], time_range=time_range)
        self.plot_ts(t, y1, z, mask, 'wspd_avg', ax=axes[2, 1], time_range=time_range)
        self.plot_ts(t, y2, z, mask, 'wspd_avg', ax=axes[3, 1], time_range=time_range)


        ### hide the other four panels for now
        axes[1, 0].axis('off')
        #axes[2, 0].axis('off')
        axes[3, 0].axis('off')
        axes[1, 2].axis('off')
        axes[2, 2].axis('off')
        axes[3, 2].axis('off')

        return mask
