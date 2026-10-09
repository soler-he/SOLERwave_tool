import numpy as np
import astropy.units as u
import sunpy.map
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from SOLERwave_find_fit_great_arcs import find_coord_from_angles


from astropy.coordinates import SkyCoord
from SOLERwave_find_fit_great_arcs import find_segments_from_list
from sunpy.coordinates import Heliocentric

import time as tm
import os



def Telescope_Instrument_string(map_base):
    '''
    Creates the name strings corresponding to an observatory/instrument
    :param map_base: reference sunpy map with full fits header
    :return: string: Custom string for the name of the observatory/instrument
    '''
    #Telescope_Instrument_string = map_base.fits_header['TELESCOP']+'/'+map_base.fits_header['INSTRUME']
    #Telescope_Instrument_string = map_base.fits_header['OBSRVTRY'] + '/' + map_base.fits_header['DETECTOR']
    # Added to fix problems with older Stereo Data not having a telescop keyword
    try:
        telescope = map_base.fits_header['TELESCOP']
    except:
        telescope = map_base.fits_header['OBSRVTRY']

    if telescope == 'STEREO':
        telescope = map_base.fits_header['OBSRVTRY']

    match telescope:
        case 'STEREO_A':
            Name_string = 'STEREO_A/EUVI'
            wave_length_string = '%.0f' % map_base.wavelength.to_value('AA')
        case 'STEREO_B':
            Name_string = 'STEREO_B/EUVI'
            wave_length_string = '%.0f' % map_base.wavelength.to_value('AA')
        case 'SDO/AIA':
            Name_string = 'SDO/AIA'
            wave_length_string = '%.0f' % map_base.wavelength.to_value('AA')
        case 'NSO-GONG':
            Name_string = 'GONG/H-alpha'
            wave_length_string = '6563'
        case 'KHPI':
            Name_string = 'Kanzelhoehe/H-alpha'
            wave_length_string = '6563'
        case 'SOLO/EUI/FSI':
            Name_string = 'SOLO/EUI/FSI'
            wave_length_string = '%.0f' % map_base.wavelength.to_value('AA')

    string = Name_string + ' ' + wave_length_string + ' $\mathrm{\AA}$ '

    return string


def full_disk_tx_ty(m_ref_height):
    '''
    Gives back the lower_Tx_Ty_lim and the Tx_Ty_range to frame the sun fully for a given instrument.
    :param m_ref_height: reference sunpy map with full fits header
    :return:
        lower_Tx_Ty_lim:   list, [lower_tx, lower_ty] in arc-seconds
        Tx_Ty_range:       float, range added to tx and ty to get to upper right corner in arc-seconds
    '''
    match m_ref_height.fits_header.get('TELESCOP'):
        case 'SDO/AIA':
            lower_Tx_Ty_lim = [-1200, -1200]
            Tx_Ty_range = 2400
        case 'STEREO':
            r_sun_observed_arcs = m_ref_height.fits_header.get('RSUN')
            lower_Tx_Ty_lim = [-r_sun_observed_arcs * 1.3, -r_sun_observed_arcs * 1.3]
            Tx_Ty_range = r_sun_observed_arcs * 2.6
        case 'SOLO/EUI/FSI':
            r_sun_observed_arcs = m_ref_height.fits_header.get('RSUN_OBS')
            lower_Tx_Ty_lim = [-r_sun_observed_arcs * 1.3, -r_sun_observed_arcs * 1.3]
            Tx_Ty_range = r_sun_observed_arcs * 2.6
        case 'KHPI':
            r_sun_observed_arcs = m_ref_height.fits_header.get('RSUN_ARC')
            lower_Tx_Ty_lim = [-r_sun_observed_arcs * 1.3, -r_sun_observed_arcs * 1.3]
            Tx_Ty_range = r_sun_observed_arcs * 2.6
        case 'NSO-GONG':
            r_sun_observed_arcs = m_ref_height.fits_header.get('SOLAR-R')
            lower_Tx_Ty_lim = [-r_sun_observed_arcs * 1.3, -r_sun_observed_arcs * 1.3]
            Tx_Ty_range = r_sun_observed_arcs * 2.6
        case _:
            assert False, 'Instrument is not jet implemented in the SOLERwave_plotting_tool/full_disk_tx_ty function'
    return lower_Tx_Ty_lim[0],lower_Tx_Ty_lim[1], Tx_Ty_range


def oktant_plot(m_data_list,
                base_map,
                flare_coordinates,
                r_sun_ref,
                m_ref_height,
                ttheta,
                aangles_along_arc,
                time,
                file_path_dict = [],
                save_path = [],
                filename_appendix = [],
                j_map_range=[0.75,1.25],
                flat_field = False):
    '''Produces an overview plot with 8 segment plots pointing N,NW,W,SW,S,SE,E,NE

    :param m_data_list:         list of 2d np.array, map sequence to be evaluated [time][py,px]
    :param base_map:            sunpy.map.Map object of the reference fits
    :param flare_coordinates:   Skycord object with the coordinates of the flare
    :param r_sun_ref:           float, reference_height_parameter in "distance" astropy.unit
    :param m_ref_height:        reference sunpy map with full fits header
    :param ttheta:              2d np.array, see pixel_to_great_segment function
    :param aangles_along_arc:   2d np.array, see pixel_to_great_segment function
    :param time:                1d list, contains the .data.value() strings of the non derotated maps
    :param j_map_range:         list with 2 float values, upper and lower limit of the j_map range, default [0.75,1.25]
    :return: -
    '''

    from sunpy.time import parse_time
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator

    time_sunpyobj = parse_time(time)
    t_sunpy_sec = (time_sunpyobj - time_sunpyobj[0]).to_value('sec')
    time_dateobj = np.array(time, dtype='datetime64[ns]')

    if flat_field:
        min_dist_to_edge = np.min([np.min(aangles_along_arc[0,:]),
                                   np.min(aangles_along_arc[-1,:]),
                                   np.min(aangles_along_arc[:,0]),
                                   np.min(aangles_along_arc[:,-1])])
        # Distance along plane in Mega Meter
        #angles_along_arc_range = np.arange(0, min_dist_to_edge-20, 12.1)
        angles_along_arc_range = np.arange(0, 700, 12.1)
    else:
        # Angles along arc in Degree
        angles_along_arc_range = np.linspace(0,np.pi*90/180,90)
    theta_range = np.linspace(-np.pi/8,15/8*np.pi,9)

    nr_of_segments = (len(angles_along_arc_range)-1) * (len(theta_range)-1)

    ##############################evaluate Data ######################################################################

    map_base = base_map

    intensity_mean,intensity_median,intensity_var,pixel_per_segment,mask_3 = find_segments_from_list(theta_range,
                                                                                                     ttheta,
                                                                                                     angles_along_arc_range,
                                                                                                     aangles_along_arc,
                                                                                                     m_data_list)
    if flat_field:
        distance = (angles_along_arc_range[:-1] + np.diff(angles_along_arc_range) / 2)

    else:
        distance = (angles_along_arc_range[:-1] + np.diff(
            angles_along_arc_range) / 2) * r_sun_ref

        distance = distance.to_value('Mm') # in Mega Meter


    #If instrument should be automatically inserted:
    # map_base.instrument + ' : ' + ....
    plot_title =Telescope_Instrument_string(map_base)+' : '+  map_base.date.value[:-4]+' UT' + '\n'+' Wave Origin: Longitude %.0f' % (flare_coordinates.Tx.to_value()) +u'\u2033' + ', Latitude: %.0f' %( flare_coordinates.Ty.to_value())+u'\u2033'

    fig= plt.figure(figsize=(11, 11))#, gridspec_kw={'wspace': 0.1, 'hspace': 0.5})
    fig.suptitle(plot_title,y = 0.96)

    fig.autofmt_xdate()

    #numberation = [[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]]
    numberation = [333,336,339,338,337,334,331,332]

    # Extract the minutes from the time string
    #time_start = int(time[0][14:16])
    #time_end = int(time[-1][14:16])
    time_start = t_sunpy_sec[0]
    time_end = t_sunpy_sec[-1]


    ax = [[] for i in range(8)]

    # Plot the 8 stack plots
    for j in range(8):

        #intensity_slice_mean = intensity_mean[:,j,:]
        intensity_slice_median = intensity_median[:, j, :]

        theta_angle = (theta_range[j] + theta_range[j + 1]) / 2

        ax[j] = fig.add_subplot(numberation[7-j])

        image_array = intensity_slice_median[::-1,:]

        image_array_time_adjusted = np.zeros((image_array.shape[0],int(time_end)))

        for index_t in range(len(t_sunpy_sec)-1):
            image_array_time_adjusted[:,int(t_sunpy_sec[index_t]):int(t_sunpy_sec[index_t+1])] =\
                np.einsum('i,j->ij',image_array[:,index_t],np.ones(int(t_sunpy_sec[index_t+1]-int(t_sunpy_sec[index_t]))))

        #https://stackoverflow.com/questions/23139595/dates-in-the-xaxis-for-a-matplotlib-plot-with-imshow
        #im = ax[j].imshow(image_array, interpolation='none',norm=colors.Normalize(vmin=0.75, vmax=1.25),extent=[time_start,time_end,distance[0],distance[-1]])
        im = ax[j].imshow(image_array_time_adjusted, interpolation='none', norm=colors.Normalize(vmin=j_map_range[0], vmax=j_map_range[1]),
                          extent=[time_dateobj[0], time_dateobj[-1], distance[0], distance[-1]])

        ax[j].set_title('direction %.f° ' % (theta_angle*180/np.pi))

        # Add x and y axis only to the outer plots
        if np.any (j == np.array([1,2,3])):
            ax[j].set_ylabel('distance / Mm')
        #if np.any(j == np.array([3,4,5])):
        #    ax[j].set_xlabel('time-t_initial / sec')

        aspect = (time_end - time_start)/(distance[-1]-distance[0]) # Calculate aspect ratio to be always square

        #ax[j].set_aspect(aspect)
        ax[j].set_aspect("auto")
        #fig.autofmt_xdate()

        locator = mdates.AutoDateLocator(minticks=2, maxticks=7)
        ax[j].xaxis.set_major_locator(locator)
        ax[j].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[j].xaxis.set_ticklabels([])

        if np.any(j == np.array([3,4,5])):

            #locator = mdates.AutoDateLocator(minticks=2, maxticks=5)
            formatter = mdates.ConciseDateFormatter(locator)
            #ax[j].xaxis.set_major_locator(locator)
            ax[j].xaxis.set_major_formatter(formatter)

            # https://matplotlib.org/3.4.3/gallery/ticks_and_spines/major_minor_demo.html
            #ax[j].xaxis.set_minor_locator(AutoMinorLocator(4))


    # https://stackoverflow.com/questions/41428442/horizontal-colorbar-over-2-of-3-subplots
    plt.draw()


    p0 = ax[3].get_position().get_points().flatten()
    #p1 = ax[4].get_position().get_points().flatten()
    p2 = ax[5].get_position().get_points().flatten()
    #ax_cbar = fig.add_axes([p0[0], 0.05, p2[2] - p0[0], 0.015])
    ax_cbar = fig.add_axes([p0[0], 0.13, p2[2] - p0[0], 0.015]) #(left, bottom, width, height)
    plt.colorbar(im, cax=ax_cbar, orientation='horizontal')

    ########### Plot central star ##################

    # Add numbers to the regions
    # https://docs.sunpy.org/en/stable/generated/gallery/plotting/magnetogram_active_regions.html#sphx-glr-generated-gallery-plotting-magnetogram-active-regions-py
    #

    lower_Tx_lim,lower_Ty_lim, Tx_Ty_range = full_disk_tx_ty(m_ref_height)
    Tx_lim = [lower_Tx_lim, lower_Tx_lim+ Tx_Ty_range]
    Ty_lim = [lower_Ty_lim, lower_Ty_lim + Tx_Ty_range]

    roi_bottom_left = SkyCoord(Tx=Tx_lim[0] * u.arcsec, Ty=Ty_lim[0] * u.arcsec, frame=m_ref_height.coordinate_frame)
    roi_top_right = SkyCoord(Tx=Tx_lim[1] * u.arcsec, Ty=Ty_lim[1] * u.arcsec, frame=m_ref_height.coordinate_frame)

    ylim_low,xlim_low = m_ref_height.wcs.world_to_array_index(roi_bottom_left)
    ylim_high,xlim_high = m_ref_height.wcs.world_to_array_index(roi_top_right)

    mask_2_plot = (mask_3 // (len(angles_along_arc_range)-1)) + 2+1000 #the + 1000 is to destinguise the levels stronger from the 0 background
    mask_2_plot[mask_3 < 1] = 0

    #map_unit = u.dimensionless_unscaled
    map_unit = map_base.quantity.unit

    m_t_1 = sunpy.map.Map(mask_2_plot *map_unit, map_base.fits_header)
    ax2 = fig.add_subplot(335,projection=map_base)
    m_t_1.draw_contours(levels=(np.arange(0,10)+1000)* map_unit,axes=ax2)

    map_base.plot(axes=ax2)#, cmap='inferno', clip_interval=(1, 99.5) * u.percent)
    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_title('Reference map: '+  map_base.date.value[:-4]+' UT')

    ax2.set_xlim([xlim_low, xlim_high])
    ax2.set_ylim([ylim_low, ylim_high])

    for theta_angle in ((theta_range[:-1] + theta_range[1:]) / 2):
        start = flare_coordinates.transform_to(Heliocentric)

        textbox_coord = SkyCoord(r_sun_ref * 1.1 * -1*np.sin(theta_angle),
                                 r_sun_ref * 1.1 * np.cos(theta_angle),
                                 0 * u.m,
                                 obstime=start.obstime,
                                 observer=start.observer,
                                 frame=Heliocentric)

        x_d_marker, y_d_marker = m_ref_height.wcs.world_to_pixel(textbox_coord)

        plt.text(x_d_marker,
                 y_d_marker,
                 ' %.f° ' % (theta_angle * 180 / np.pi),
                 horizontalalignment='center',
                 verticalalignment='center',
                 # fontsize=font_size * 3 / 4,
                 color='white',
                 rotation = theta_angle * 180 / np.pi)

    #map_base.draw_limb(axes=ax2, color='C0')

    # How to print text in the plots if necassary
    #transparent_white = (1, 1, 1, 0.5)
    #ax2.text(200,200,'test',color='black',fontweight='bold',backgroundcolor=transparent_white)

    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        save_path = file_path_dict['path_LVL_0_Results_0']
        filename_appendix = file_path_dict['filename_appendix']
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Octant Plot : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                    'Only the file_path_dict was used')
    elif len(file_path_dict) != 0:
        save_path = file_path_dict['path_LVL_0_Results_0']
        filename_appendix = file_path_dict['filename_appendix']

    if len(filename_appendix) == 0:
        filename_appendix = ''

    if len(save_path) != 0:
        octant_path = os.path.join(save_path, 'Octantplot' + filename_appendix+'.png')
        octant_path_pdf = os.path.join(save_path, 'Octantplot' + filename_appendix + '.pdf')
        plt.savefig(octant_path,dpi = 300)
        plt.savefig(octant_path_pdf,dpi = 300, format='pdf')

        from PIL import Image
        im = Image.open(octant_path)
        width, height = im.size

        # Setting the points for cropped image
        left = 60
        top = 40
        right = width-90
        bottom = height-110

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        im1.save(octant_path)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + ' Octant Plot: finished')

def lineplot_with_subplots(distance,
                           intensity_mean,
                           intensity_std,
                           time,
                           plot_title,
                           additional_plot = np.array([]),
                           additional_distance = np.array([]),
                           nr_of_colums = 4,
                           nr_of_lines = [],
                           x_lim = [],
                           y_min = [],
                           y_max = [],
                           max_values_distance =[],
                           max_values=np.array([]) ,
                           uncertainty_distance = np.array([]),
                           uncertainty_height = np.array([]),
                           plot_in_multiple_figures = False,
                           order_by_line = True,
                           adplot_kwargs = {'color':'green', 'marker':'o', 'linestyle':'dashed'},
                           max_kwargs = {'color':'red', 'marker':'x'},
                           file_path_dict = [],save_path = [],filename_appendix = []):
    ''' Generalized function to plot time dependent line plots in multiple subplots. Allows the overprinting of additional
    lines, e.g. the lines filtered. SOLERwave uses this function for the creating of its Perturbation plots.

    :param distance:        1d np.array, distance corresponding to the intesity values
    :param intensity_mean:  2d np.array, mean intensity value coresponding to [distance, time]
    :param intensity_std:   2d np.array, std intensity value coresponding to [distance, time]
    :param time:            1d list,    timestrings of each observation
    :param plot_title:                  Supertitle of the Plot
    :param additional_plot: optional,3d np.array, additional values to plot in the form [(add_)distance,time,nr_ad_plot]
    :param additional_distance: optional,2d or 3d np.array, distance values for additional plot(s) [add_distance,(time),nr_ad_plot]
                                                        If no value is given, distance will be used as default
    :param nr_of_colums:    int ,       number of colums
    :param nr_of_lines:     int ,       number of lines, if not given the total number of timesteps will be used
    :param x_lim:           1d np.array, lower and upper x_lim in the form [x_lower, x_upper]
    :param y_min:           float,      lower y limit, default is 0.9 times the minimum intensity - std
    :param y_max:           float,      upper y limit, default is 1.05 times the maximum intensity + std
    :param max_values_distance  2d np.array, distance of the maxium values, [time,max_values]
    :param max_values:          2d np.array, maximum values, [time,max_values]
    :param uncertainty_distance 1d np.array
    :param uncertainty_height   1d np.array
    :param max_additiv:         float,   value to be added to the maximum values, default = 0
    :param plot_in_multiple_figures: bool, plot in multiple figures or end after the first
    :param order_by_line:       bool,    order the timesteps in lines, default is True
    :param adplot_kwargs:       dict,    additional keyword for the added plots
    :param max_kwargs:          dict,   additional keyword for the maximum marker
    :param file_path_dict:      dict, holds default file paths
    :param save_path:           string,custom file path, overruled by file_path_dict if both are given
    :param filename_appendix:   string, custom filename_appendix, overruled by file_path_dict if both are given
    :return:
    '''

    #########################################
    # Following extracts the saving data from the file_path_dict and warns the user if both a save_path
    # and file path dict is given
    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        save_path = file_path_dict['path_LVL_0_Results_0_Diagnostics']
        filename_appendix = file_path_dict['filename_appendix']
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(
            now + ' plot_perturbation_profiles/lineplot_with_subplots : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                  'Only the file_path_dict was used')
    elif len(file_path_dict) != 0:
        save_path = file_path_dict['path_LVL_0_Results_0_Diagnostics']
        filename_appendix = file_path_dict['filename_appendix']

    if len(filename_appendix) == 0:
        filename_appendix = ''
    ##########################################

    if additional_plot.size != 0:
        mult_ad_plots = len(additional_plot.shape) == 3
        if mult_ad_plots:
            nr_of_add_plots = additional_plot.shape[2]

    if additional_distance.size == 0:
        additional_distance = distance

    #distance = distance / (1e6 * u.meter)
    #additional_distance = additional_distance/ (1e6*u.meter)

    if nr_of_lines:
        l_i = nr_of_lines
        t_range = [range(nr_of_colums*l_i)]

        if nr_of_colums*l_i < intensity_mean.shape[1]:

            if plot_in_multiple_figures:
                nr_of_plots = intensity_mean.shape[1] // (nr_of_colums*l_i)
                t_rest = intensity_mean.shape[1] % (nr_of_colums*l_i)
                t_range =[range(nr_of_colums*l_i) for _ in range(nr_of_plots)]

                # Adds a plot if there are any rest plots
                if t_rest != 0:
                    nr_of_plots = nr_of_plots + 1
                    t_range.append(range(t_rest))

            else:
                now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                print(now + ' Lineplot_with_subplots: Warning: not all time steps are plotted in the lineplot. Set plot_in_multiple_figures = true to plot all')


        elif nr_of_colums*l_i >= intensity_mean.shape[1]:
            t_range = [range(intensity_mean.shape[1])] #Makes sure nothing gets indexed that is not available
            nr_of_plots = 1
    else:
        l_i = int(np.ceil(intensity_mean.shape[1]/nr_of_colums))
        t_range = [range(intensity_mean.shape[1])]

    # Sets all y_limits to the same value
    if not y_max:
        y_max = 1.05 * np.max(intensity_mean + intensity_std)
    if not y_min:
        y_min = 0.9 * np.min(intensity_mean - intensity_std)

    for plot_index in range(nr_of_plots):
        fig, ax = plt.subplots(l_i,nr_of_colums,figsize=(18, 12),gridspec_kw={'wspace': 0.2, 'hspace': 0.5}) #sharex=True,sharey = True

        fig.suptitle(plot_title + f'figure:{plot_index+1}/{nr_of_plots}',y = 0.95)

        for t in t_range[plot_index]:
            #for j in range(2):


            if order_by_line:
                i = t // nr_of_colums
                j = t % nr_of_colums
            else:
                i = t % l_i
                j = t//l_i

            # Added to allow for plotting of multiple figures from one dataset
            t = t+plot_index*nr_of_colums*l_i

            #prominences = peak_prominences(intensity_mean[:, t], max_index)[0]

            ax[i,j].plot(distance,intensity_mean[:, t], '-b')#,markersize=2)
            ax[i,j].plot([distance[0],distance[-1]],[1,1],'--r')
            #ax[i,j].plot(distance[max_index],prominences+1,'-xk')

            #x = np.arange(intensity_mean.shape[1])
            ax[i,j].fill_between(distance,intensity_mean[:, t] + intensity_std[:, t],intensity_mean[:, t] - intensity_std[:, t],alpha=0.2,color='b')
            ax[i,j].grid(True)

            ax[i,j].set_title(time[t])
            ax[i,j].set_ylim([y_min,y_max])


            if max_values.size != 0:
                #max_index = max_values_index[t]
                ax[i, j].plot(max_values_distance[t,:], max_values[t,:]+1, markersize=10,**max_kwargs)

            if uncertainty_distance.size != 0:
                ax[i, j].plot(uncertainty_distance[t,:],uncertainty_height[t,:],'-xr')

            if x_lim:
                ax[i,j].set_xlim(x_lim)
            else:
                ax[i,j].set_xlim([distance[0],distance[-1]])

            # If additional lines shall be ploted
            if additional_plot.size != 0:
                # Checks if one or more additional lines shall be plotted
                if mult_ad_plots:
                    # Iterates over multiple lines
                    for k in range(nr_of_add_plots):
                        # Checks if on which distance scale they should be plotted
                        if len(additional_distance.shape) == 2:
                            ax[i, j].plot(additional_distance[:,k], additional_plot[:, t, k], **adplot_kwargs)
                        elif len(additional_distance.shape) == 3:
                            ax[i, j].plot(additional_distance[:,t,k], additional_plot[:, t, k], **adplot_kwargs)
                        else:
                            ax[i, j].plot(additional_distance, additional_plot[:, t, k], **adplot_kwargs)
                else:
                    ax[i, j].plot(additional_distance, additional_plot[:, t], **adplot_kwargs)

        for j in range(nr_of_colums):
            ax[-1,j].set_xlabel('arc distance / Mm')
        #ax[int(intensity_mean.shape[1]/2)].set_ylabel('relative intensity')

        ####################################################################
        # Added Saving capability
        ####################################################################

        if len(save_path) != 0:
            plot_path = os.path.join(save_path, f'Perturbation_Profile_{plot_index+1}_of_{nr_of_plots}' + filename_appendix + '.png')
            plt.savefig(plot_path)

#Test

def plot_perturbation_profiles(distance,
                               intensity_mean,
                               intensity_var,
                               d_peak_mat,
                               d_trail_mat,
                               d_front_mat,
                               peak_mat,
                               trail_mat,
                               front_mat,
                               time,
                               instr_dir_width_title_string,
                               sector_nr = 0,
                               linplot_ylim = [0.9, 1.6],
                               wave_ind_mat_2 = np.array([]),
                               show_all_peaks = True,
                               file_path_dict = [], save_path = [], filename_appendix = [],
                               **kwargs):
    """ Plots the perturbation profiles for a sector_nr with overlaid peak maxima, front- and trailing edge in multiple
    plots with subplots each to cover in summ (nr of plots * nr of subplots) all time steps.

    :param matrices: See generating functions of the SOLERwave tool for description
    :param time:             list with strings, time string of the observation of images
    :param wave_ind_mat_2:   3d np.array, if given only peaks associated to waves are marked, default empty np.array([])
    :param instr_dir_width_title_string: string, part of the title of the plot, see tutorial jupyter notebook of SOLERwave tool for more information
    :param sector_nr:        int, index of sector defined by theta range
    :param linplot_ylim:     1d np.array, lower and upper y_limit
    :param file_path_dict:      dict, holds default file paths
    :param save_path:           string,custom file path, overruled by file_path_dict if both are given
    :param filename_appendix:   string, custom filename_appendix, overruled by file_path_dict if both are given
    :param kwargs: catches all additional keyword arguments, allowing the function to be cast by a dict with to many
                    entries
    :return:
    """

    j = sector_nr

    single_plot_kwargs = {'nr_of_colums': 5, 'nr_of_lines': 4, 'y_min': linplot_ylim[0], 'y_max': linplot_ylim[1]}

    d_trail_mat_ = np.copy(d_trail_mat[j, :, :])
    d_front_mat_ = np.copy(d_front_mat[j, :, :])
    trail_mat_ = np.copy(trail_mat[j, :, :])
    front_mat_ = np.copy(front_mat[j, :, :])
    d_peak_mat_ = np.copy(d_peak_mat[j, :, :])
    peak_mat_ = np.copy(peak_mat[j, :, :])

    file_path_dict_ = file_path_dict.copy()

    if not show_all_peaks:
        # Sets all peaks not associated with a wave to np.nan
        d_trail_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan
        d_front_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan
        trail_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan
        front_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan
        d_peak_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan
        peak_mat_[wave_ind_mat_2[j,:,:]==0] = np.nan

        adplot_kwargs = {'color': 'green', 'marker': 'x', 'linestyle': 'dashed'}
        max_kwargs = {'color': 'cyan', 'marker': 'x','linestyle': ''}

        plot_title_perturbation = 'Only Wave-Peaks: '+ instr_dir_width_title_string + ' '
        appendix_string = '_only_wave_peaks'
    else:
        adplot_kwargs = {'color': 'green', 'marker': 'x', 'linestyle': 'dashed'}
        max_kwargs = {'color': 'red', 'marker': 'x','linestyle': ''}

        plot_title_perturbation = 'All Peaks: ' + instr_dir_width_title_string + ' '
        appendix_string = '_all_peaks'

    if 'filename_appendix' in file_path_dict_:
        file_path_dict_['filename_appendix'] = appendix_string+ file_path_dict_['filename_appendix']

    d_wave_range = np.stack((d_trail_mat_[ :, :], d_front_mat_[ :, :]), axis=0)  # d_front_mat[j,t,i]
    wave_range = np.stack((trail_mat_[ :, :], front_mat_[ :, :]), axis=0)

    max_values_distance = d_peak_mat_[ :, :]
    max_values = peak_mat_[:, :] - 1



    lineplot_with_subplots(distance.to_value(u.Mm), intensity_mean[:, j, :],np.sqrt(intensity_var[:, j, :]),
                           time,plot_title_perturbation,max_values_distance=max_values_distance,max_values=max_values,
                           plot_in_multiple_figures = True,additional_distance=d_wave_range,
                           additional_plot=wave_range,adplot_kwargs = adplot_kwargs,max_kwargs = max_kwargs,
                           file_path_dict = file_path_dict_,save_path = save_path,filename_appendix = filename_appendix
                           ,**single_plot_kwargs)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + ' Plot_pertubation_profiles: finished')
    return

def coord_for_sector(Flare_coordinates,theta_range,angles_along_arc_range,flat_field):
    '''
    Calculates the sky_coordinates of a contour around the sector specified by the first two values of the theta_range
    and the angles_along_arc_range.
    :param Flare_coordinates:       Skycord object with the coordinates of the flare
    :param theta_range:             1d np.array with 2 float values, upper and lower segment borders along theta direction
    :param angles_along_arc_range:  1d np.array, vector of the segment borders along the arc direction
    :param flat_field:              bool, switches between flat field at sun distance and following the sun curvature
    :return:
    '''

    from SOLERwave_find_fit_great_arcs import find_coord_from_angles
    # Create Theta values for the segement

    theta_vec = np.linspace(theta_range[0], theta_range[1], 20)

    theta_mat = np.concatenate([np.ones_like(angles_along_arc_range) * theta_range[0],
                                theta_vec,
                                np.ones_like(angles_along_arc_range) * theta_range[1],
                                theta_vec[::-1]])
    aaa_mat = np.concatenate([angles_along_arc_range,
                              np.ones_like(theta_vec) * angles_along_arc_range[-1],
                              angles_along_arc_range[::-1],
                              np.ones_like(theta_vec) * angles_along_arc_range[0]])

    coords_meter= find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat,flat_field)

    coords_sky = SkyCoord(coords_meter[0, :], coords_meter[1, :], coords_meter[2, :],
                            obstime=Flare_coordinates.obstime,
                            observer=Flare_coordinates.observer,
                            frame=Heliocentric).transform_to(Flare_coordinates.frame)
    return coords_sky




def plot_timeseries_of_lineplot_and_map(segment_nr,
                                        theta_range,
                                        map_series,
                                        m_ref_height,
                                        intensity_mean,
                                        intensity_var,
                                        time,
                                        t_sunpy_sec,
                                        distance,
                                        linplot_ylim,
                                        wave_peak_kwargs,
                                        segment_kwargs,
                                        str_direct_width,
                                        lower_Tx_lim=0,
                                        lower_Ty_lim=0,
                                        Tx_Ty_range=0,
                                        Tx_Ty_aspect_ratio = False,
                                        Tx_lim=[],
                                        Ty_lim =[],
                                        plot_wavepeak=True,
                                        plot_peaks = True,
                                        plot_sector=True,
                                        scale_min = 0.5,
                                        scale_max = 1.5,
                                        overplot_map = False,
                                        overplot_map_series = [],
                                        overplot_kwargs = {},
                                        file_path_dict = [],
                                        save_path = [],
                                        filename_appendix = [],
                                        distance_marker = True,
                                        distance_marker_label=False,
                                        contrast_mode = False,
                                        quicklook = False,
                                        flat_field = False,
                                        framing ='',  #Options 'auto_framing',#'full_disk'
                                        add_overlay_function = False,
                                        overlay_function = None,
                                        overlay_function_keywords = []):
    """ Creates a Movie for 1 sector with the perturbation profile on the left side and the base ratio image on the right

    :param segment_nr:      int, index of sector defined by theta range
    :param theta_range:     1d np.array, see SOLERwave_find_fit_great_arcs
    :param map_series:      sunpy.map.Map series object with all maps
    :param m_base:          reference image as sunpy.map.Map
    :param intensity_mean:  3d np.array, see SOLERwave_find_fit_great_arcs
    :param intensity_var:   3d np.array, see SOLERwave_find_fit_great_arcs
    :param time:            list of strings, see SOLERwave_find_fit_great_arcs
    :param distance:        1d np.array, Distance in length units (e.g. [1e6,2e6] * u.m) of astropy.units
    :param linplot_ylim:    1d np.array, y - limit of the lineplot (left panel)
    :param wave_peak_kwargs: dict, Quantities important for the wave plot
                                    {'d_peak_mat':d_peak_mat,'d_front_mat':d_front_mat,'theta_range':theta_range,
                                    'peak_mat':peak_mat,'front_mat':front_mat,'Flare_coordinates':Flare_coordinates,
                                    'wave_ind_mat_2':wave_value_dict['wave_ind_mat_2']}
    :param segment_kwargs: dict, Quantities important for plotting the segments
                                    {'mask_3':mask_3, 'angles_along_arc_range':angles_along_arc_range}
    :param str_direct_width: sting, part of title string with direction and width of sector of interest
    :param lower_Tx_lim:    float, lower Tx limit in arcseconds
    :param lower_Ty_lim:    float, lower Ty limit in arcseconds
    :param Tx_Ty_range:     float, frame width/height in arcseconds from lower Tx/Ty
    :param Tx_Ty_aspect_ratio: bool, if false (default), than Tx_lim and Ty_lim are calculated form
                                        [Tx_lower, Tx_lower + Tx_Ty_range] and [Ty_lower, Ty_lower + Tx_Ty_range]
    :param Tx_lim:          1d np.array, Tx limit in arcseconds
    :param Ty_lim:          1d np.array, Ty limit in arcseconds
    :param plot_wavepeak:   bool, if true plots the wave peaks
    :param plot_peaks:      bool, if true plots all peaks are plotted (wave peaks are overplotted if also true)
    :param plot_sector:     bool,  if true plots the sector outline
    :param map_keywords:    dict, keyword arguments passed to Map class, default using scale_min and scale_max
    :param scale_max:       float, maximum of scale of the base ratio image
    :param scale_min:       float, minimum of scale of the base ratio image
    :param file_path_dict:      dict, holds default file paths
    :param save_path:           string,custom file path, overruled by file_path_dict if both are given
    :param filename_appendix:   string, custom filename_appendix, overruled by file_path_dict if both are given
    :param distance_marker:     bool, if true plots the distance marker on the sector
    :param distance_marker_label: bool, if true, plots numbers equal to Mm next to the distance marker
    :param contrast_mode:       bool, if true activates contrast mode and outlines the sector plot with a white line
    :param quicklook:           bool, if true computes and shows 4 images instead of creating the movie
    :param flat_field:          bool, switches between flat field at sun distance and following the sun curvature
    :param framing:             string, either 'auto_framing', 'full_disk' or empty string '' (default)
    :return:
    """

    def gaussian(x, A, x0, sigma1):#,sigma2):
        return 1 + A * np.exp(-(x - x0) ** 2 / (2 * sigma1 ** 2))

    import matplotlib.gridspec as gridspec
    from SOLERwave_find_fit_great_arcs import find_coord_from_angles
    import time as tm

    # Converts distance to Mm
    distance = distance.to_value(u.Mm)

    line_color = 'black'
    front_color= 'green'
    peak_color = 'blue'
    wave_peak_color = 'blue'
    peak_front_marker ='X' #https://matplotlib.org/stable/api/markers_api.html
    #font_size = 35
    #wave_peak_front_linewidth = 4
    #linewidth_sector = 4
    #linewidth_pert = 4

    font_size = 24
    wave_peak_front_linewidth = 2
    linewidth_sector = 2
    linewidth_pert = 2


    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Feature Plot : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                    'Only the file_path_dict was used')  # TODO: might ues an actual waring package

    # Set Path for movie images to default or to a custom path
    if len(file_path_dict) != 0:
        path_LVL_0_Results_0_Diagnostics = file_path_dict['path_LVL_0_Results_0_Diagnostics']
        path_LVL_0_Results_0 = file_path_dict['path_LVL_0_Results_0']
        filename_appendix = file_path_dict['filename_appendix']
    else:
        path_LVL_0_Results_0_Diagnostics = save_path
        path_LVL_0_Results_0 = save_path

    # Create a movie image folder based on if the plot is diagnostic or general
    if plot_peaks:
        path_LVL_0_Results_0_Diagnostics_MI = os.path.join(path_LVL_0_Results_0_Diagnostics,'Movie_Images_all_Peaks')
    else:
        path_LVL_0_Results_0_Diagnostics_MI = os.path.join(path_LVL_0_Results_0_Diagnostics,'Movie_Images_Kinematics')
    os.makedirs(path_LVL_0_Results_0_Diagnostics_MI, exist_ok=True)

    # Remove any files in the folder before creating new ones
    try:
        for file in os.listdir(path_LVL_0_Results_0_Diagnostics_MI):
            file_path = os.path.join(path_LVL_0_Results_0_Diagnostics_MI,file)
            os.remove(file_path)
    except:
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Plotting_tools: plot_timeseries_of_lineplot_and_map: problem with filepath')


    overplot_info=''
    if overplot_map:
        overplot_info = overplot_info + '_map_overplotted'
    if add_overlay_function:
        overplot_info = overplot_info + '_overlaid_function'



    j = segment_nr #Staying concistent with using j for numerating the theta angle

    intensity_slice_mean = intensity_mean[:, j, :]
    intensity_slice_std = np.sqrt(intensity_var[:, j, :])

    map_unit = u.dimensionless_unscaled

    if plot_sector:
        #mask_3 = segment_kwargs['mask_3']
        angles_along_arc_range = segment_kwargs['angles_along_arc_range']
        Flare_coordinates = wave_peak_kwargs['Flare_coordinates']

        contours1 = coord_for_sector(Flare_coordinates, theta_range, angles_along_arc_range,flat_field)

        #mask_2_plot = (((mask_3 -1)// (len(angles_along_arc_range)-1)))+1
        #mask_2_plot[mask_3 < 1] = 0
        #mask_2_plot[mask_2_plot != (j+1)] = 0

        #m_t_1 = sunpy.map.Map(mask_2_plot * map_unit, map_series[0].fits_header)
        #contours1 = m_t_1.contour(0.5 * u.dimensionless_unscaled)

    if plot_wavepeak or plot_peaks:
        d_peak_mat = wave_peak_kwargs['d_peak_mat']
        peak_mat = wave_peak_kwargs['peak_mat']
        d_front_mat = wave_peak_kwargs['d_front_mat']
        front_mat = wave_peak_kwargs['front_mat']
        d_trail_mat = wave_peak_kwargs['d_trail_mat']
        trail_mat = wave_peak_kwargs['trail_mat']
        Flare_coordinates = wave_peak_kwargs['Flare_coordinates']
        wave_ind_mat_2 = wave_peak_kwargs['wave_ind_mat_2']

        plot_fitted_peak = wave_peak_kwargs['plot_fitted_peak']
        if plot_fitted_peak:
            sig_fitted_peak_mat = wave_peak_kwargs['sig_fitted_peak_mat']

        if flat_field:
            d_peak_mat_aaa = d_peak_mat
            d_front_mat_aaa = d_front_mat
        else:
            d_peak_mat_aaa = d_peak_mat*1e6/(m_ref_height.fits_header['rsun_ref']) #Convert to angle along arc
            d_front_mat_aaa = d_front_mat*1e6/(m_ref_height.fits_header['rsun_ref']) #Convert to angle anlong arc

        max_values_distance = d_peak_mat[j,:,:]
        front_values_distance = d_front_mat[j,:,:]
        max_values = peak_mat[j,:,:]
        front_values = front_mat[j,:,:]
        trail_values = trail_mat[j,:,:]
        trail_values_distance = d_trail_mat[j,:,:]


    #####################################################
    # Limits for the Map Plot
    #####################################################
    if framing.lower() == 'auto_framing':
        auto_framing = True
    elif framing.lower() == 'full_disk':
        auto_framing = False
        # Overwrite any giver values with the full disk ranges
        lower_Tx_lim,lower_Ty_lim, Tx_Ty_range = full_disk_tx_ty(m_ref_height)
    else:
        auto_framing = False

    # If no range is given and no Tx_lim, use full width
    if (Tx_Ty_range == 0) & (len(Tx_lim)==0):
        Tx_lim = [m_ref_height.bottom_left_coord.Tx.value, m_ref_height.top_right_coord.Tx.value]
    # If no range is given and no Ty_lim, use full width
    if (Tx_Ty_range == 0) & (len(Ty_lim)==0):
        Ty_lim = [m_ref_height.bottom_left_coord.Ty.value, m_ref_height.top_right_coord.Ty.value]

    if not Tx_Ty_aspect_ratio:
        Tx_lim = [lower_Tx_lim,lower_Tx_lim+Tx_Ty_range]
        Ty_lim = [lower_Ty_lim,lower_Ty_lim+Tx_Ty_range]

    if plot_sector & auto_framing:
        additional_area_in_arcsec = 150
        # The tx,ty values of the sector in arc seconds
        tx_vals = contours1.Tx.value
        ty_vals = contours1.Ty.value

        # Add the Flare coordinates in arcsec to the arrays
        tx_vals = np.append(tx_vals, Flare_coordinates.Tx.value)
        ty_vals = np.append(ty_vals, Flare_coordinates.Ty.value)

        # Set a 100 arcsec Border around area of interest
        Tx_lim_needed = [np.nanmin(tx_vals)-additional_area_in_arcsec,np.nanmax(tx_vals)+additional_area_in_arcsec]
        Tx_lim_diff = np.diff(Tx_lim_needed)[0]

        Ty_lim_needed = [np.nanmin(ty_vals)-additional_area_in_arcsec,np.nanmax(ty_vals)+additional_area_in_arcsec]
        Ty_lim_diff = np.diff(Ty_lim_needed)[0]

        if Tx_lim_diff > Ty_lim_diff:
            # Add Area to Ty_lim to make the aspect ratio a square
            Ty_lim_needed[0] = Ty_lim_needed[0]-((Tx_lim_diff-Ty_lim_diff)/2)
            Ty_lim_needed[1] = Ty_lim_needed[1]+((Tx_lim_diff-Ty_lim_diff)/2)
        elif Tx_lim_diff < Ty_lim_diff:
            # Add Area to Tx_lim to make the aspect ratio a square
            Tx_lim_needed[0] = Tx_lim_needed[0] - ((Ty_lim_diff - Tx_lim_diff) / 2)
            Tx_lim_needed[1] = Tx_lim_needed[1] + ((Ty_lim_diff - Tx_lim_diff) / 2)

        Tx_lim = Tx_lim_needed
        Ty_lim = Ty_lim_needed

    roi_bottom_left = SkyCoord(Tx=Tx_lim[0] * u.arcsec, Ty=Ty_lim[0] * u.arcsec, frame=m_ref_height.coordinate_frame)
    roi_top_right = SkyCoord(Tx=Tx_lim[1] * u.arcsec, Ty=Ty_lim[1] * u.arcsec, frame=m_ref_height.coordinate_frame)

    ylim_low,xlim_low = m_ref_height.wcs.world_to_array_index(roi_bottom_left)
    ylim_high,xlim_high = m_ref_height.wcs.world_to_array_index(roi_top_right)


    #####################################################

    theta_angle = (theta_range[j] + theta_range[j + 1]) / 2


    # In case of a quicklook only 4 images are produced
    if quicklook:
        time_index_range = np.linspace(0,len(time)-1,4,dtype=int)

        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' plot_timeseries_of_lineplot_and_map: Quick Look, only 4 images are produces. Rerun with quicklook == False to create a movie')
    else:
        time_index_range = range(len(time))

    # Formating and creating a progress bar
    # https://builtin.com/software-engineering-perspectives/python-progress-bar:
    from tqdm import tqdm
    import sys
    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    text = now + ' plot_timeseries_of_lineplot_and_map: Frames for Movie :'

    for t in tqdm(time_index_range,desc = text,file=sys.stdout):
        #print(f'Video Frame {t} of {len(time)}')


        #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
        fig = plt.figure(figsize=(20, 9))
        #fig.suptitle(plot_title)
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, wspace=0.2, hspace=0.3)

        ##############################################
        # Plotting of the Map
        #############################################

        ax = fig.add_subplot(spec2[0, 1], projection=map_series[t])
        im = map_series[t].plot(axes=ax, cmap='inferno',interpolation = 'none',norm=colors.Normalize(vmin=scale_min, vmax=scale_max))
        ax.set_xlim([xlim_low, xlim_high])
        ax.set_ylim([ylim_low, ylim_high])
        #map_series[t].draw_limb(axes=ax)
        #ax.set_title(m_base.instrument + '  ' + m_base.wavelength.to_string()[:5] +' $\\text{\AA}$ '+ time [t],size=20)

        # Over-plots the map with a custom other map
        if overplot_map == True:

            # Range to nicely overplot SOLO/EUI/FSI 174
            '''
            from astropy.visualization import ImageNormalize, SqrtStretch
            norm = ImageNormalize(
                vmin=50,
                vmax=2000, #500 for average, 5000 for only loops
                stretch=SqrtStretch()
            )
            overplot_map_series[t].plot(axes=ax, norm=norm, cmap='inferno')
            #'''
            overplot_map_series[t].plot(axes=ax, **overplot_kwargs)

        ax.set_title(
            Telescope_Instrument_string(m_ref_height) + ' ' + str(np.array(time[t], dtype='datetime64[s]')) + ' UT',
            size=font_size)  # ' \\text{$\AA$}
        m_ref_height.draw_limb(axes=ax,color = 'black')

        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()

        ax.set_xlabel(x_label,size=font_size)
        ax.set_ylabel(y_label,size=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size*4/5)

        if add_overlay_function:
            overlay_function(ax,t,**overlay_function_keywords)
            overplot_info = overplot_info + '_overlaid_function'

        if distance_marker:
            vec1 = np.linspace(0,1000,11)
            vec1 = vec1[vec1 > distance[0]] # all 100 mM increments below are excepted
            vec1 = vec1[vec1 < distance[-1]] # all 100 mM increment above range are excepted
            if flat_field:
                aaa_100mM = vec1
            else:
                aaa_100mM = vec1*1e6/(m_ref_height.fits_header['rsun_ref'])

            # Theta values: -1° + T_j, T_j, T_j+1, T_j+1 + 1°
            theta_angle_range = np.array([theta_range[j],theta_range[j], theta_range[j + 1],theta_range[j + 1]])
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones_like( aaa_100mM))

            if flat_field:
                theta_mat[0,:] = theta_mat[0,:] - np.pi/180 * 10*100/aaa_100mM
                theta_mat[3,:] = theta_mat[3,:] + np.pi/180 * 10*100/aaa_100mM
            else:
                theta_mat[0,:] = theta_mat[0,:]  -np.pi/180 * 1/np.sin(aaa_100mM)
                theta_mat[3,:] = theta_mat[3,:] + np.pi/180 * 1/np.sin(aaa_100mM)
            aaa_mat = np.einsum('i,j->ij', np.ones_like(theta_angle_range), aaa_100mM)

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)
            for i in range(len(vec1)):
                if contrast_mode:
                    ax.plot_coord(coords_sky[:2, i], '-w',linewidth = linewidth_sector*2)
                    ax.plot_coord(coords_sky[-2:, i], '-w',linewidth = linewidth_sector*2)
                ax.plot_coord(coords_sky[:2,i], '-k',linewidth = linewidth_sector)
                ax.plot_coord(coords_sky[-2:, i], '-k',linewidth = linewidth_sector)

            ###################################
            # Adds Text to the Distance Marker
            ###################################
            if distance_marker_label:
                if flat_field:
                    theta_vec = theta_mat[0, :]-np.pi/180 * 30*100/aaa_100mM
                else:
                    theta_vec = theta_mat[0,:]-np.pi / 180 * 3/np.sin(aaa_100mM)
                coords_meter = find_coord_from_angles(Flare_coordinates, theta_vec, aaa_100mM,flat_field)

                coords_sky = SkyCoord(coords_meter[0, :], coords_meter[1, :], coords_meter[2, :],
                                      obstime=Flare_coordinates.obstime,
                                      observer=Flare_coordinates.observer,
                                      frame=Heliocentric).transform_to(Flare_coordinates.frame)

                y_d_marker, x_d_marker = m_ref_height.wcs.world_to_array_index(coords_sky)

                for index in range(len(vec1)):
                    plt.text(x_d_marker[index],
                             y_d_marker[index],
                             '%.0f'%(vec1[index]),
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize = font_size*3/4,
                             color = 'white')

        if plot_sector:

            if contrast_mode:
                ax.plot_coord(contours1, color='white',linewidth = linewidth_sector)
            ax.plot_coord(contours1, color='black',linewidth = linewidth_sector)


        if plot_peaks:
            #theta_angle_range = theta_angle #(theta_range[:-1] + theta_range[1:]) / 2
            theta_angle_range = np.linspace(theta_range[j],theta_range[j+1], 10)
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones(d_peak_mat_aaa.shape[2]))
            #theta_vec = theta_angle_range* np.ones(d_peak_mat_aaa.shape[2])

            ###############################################################################################
            # Search for the coardinates of the Peak and the Peak + 1 Sigma
            ################################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i,:] = d_peak_mat_aaa[j, t, :]

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color = peak_color,linewidth = wave_peak_front_linewidth)
            #################################################################################################
            aaa_mat_front = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_front[i, :] = d_front_mat_aaa[j, t, :]

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_front,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color =front_color,linewidth = wave_peak_front_linewidth)

        plot_limb = False #Todo: create the limb distance range automatically
                          #      make the lines thinner, so they don't disturb
        if flat_field & plot_limb:
            #theta_angle_range = theta_angle #(theta_range[:-1] + theta_range[1:]) / 2
            limb_distance = np.array([400,500])
            theta_angle_range = np.linspace(theta_range[j],theta_range[j+1], 10)
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones_like(limb_distance))
            #theta_vec = theta_angle_range* np.ones(d_peak_mat_aaa.shape[2])

            ###############################################################################################
            # Search for the coardinates of the Peak and the Peak + 1 Sigma
            ################################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i,:] = limb_distance

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color = 'black',linewidth = wave_peak_front_linewidth)




        if plot_wavepeak:
            d_waves_peak_aaa = d_peak_mat_aaa[j,t,:][wave_ind_mat_2[j,t,:] != 0] #Only peaks part of waves are named
            d_waves_front_aaa = d_front_mat_aaa[j,t,:][wave_ind_mat_2[j,t,:] != 0] #Only peaks part of waves are named

            theta_angle_range = np.linspace(theta_range[j],theta_range[j+1], 10)
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones_like(d_waves_peak_aaa))
            #theta_vec = theta_angle_range* np.ones(d_peak_mat_aaa.shape[2])

            ###############################################################################################
            # Search for the coardinates of the Peak and the Peak + 1 Sigma
            ################################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i,:] = d_waves_peak_aaa

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color = wave_peak_color,linewidth = wave_peak_front_linewidth)

            ############################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i, :] = d_waves_front_aaa

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak,flat_field)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--', color=front_color,linewidth = wave_peak_front_linewidth)


        ####################################################
        # Plotting the lines
        ####################################################
        ax2 = fig.add_subplot(spec2[0, 0])

        if plot_peaks:
            if plot_fitted_peak:
                for peak_nr in range(len(max_values[t, :])):
                    fit_vec = np.linspace(trail_values_distance[t,peak_nr],front_values_distance[t,peak_nr],100)
                    gauss_fit = gaussian(fit_vec, max_values[t,peak_nr]-1, max_values_distance[t, peak_nr], sig_fitted_peak_mat[j,t,peak_nr])
                    ax2.plot(fit_vec, gauss_fit, '-r',linewidth =linewidth_pert)


        ax2.plot(distance, intensity_slice_mean[:, t], '-',color = line_color,linewidth =linewidth_pert)  # ,markersize=2)
        ax2.plot([distance[0], distance[-1]], [1, 1], '--r',linewidth =linewidth_pert)

        # Plot Limb Distance
        if flat_field & plot_limb:
            ax2.plot([limb_distance[0],limb_distance[0]],[linplot_ylim[0], linplot_ylim[1]],'--k',linewidth =linewidth_pert)
            ax2.plot([limb_distance[1], limb_distance[1]], [linplot_ylim[0], linplot_ylim[1]], '--k',linewidth=linewidth_pert)
        #y_max = 1.2 #1.05 * np.max(intensity_slice_mean)
        #y_min = 0.9 #0.9 * np.min(intensity_slice_mean)

        ax2.set_ylim([linplot_ylim[0], linplot_ylim[1]])
        aspect = (linplot_ylim[1]- linplot_ylim[0])/(distance[-1]- distance[0])
        ax2.set_aspect(1/aspect)
        ax2.set_xlabel('Distance (Mm)',size=font_size)
        ax2.set_ylabel('Amplitude',size=font_size)
        ax2.tick_params(axis='both', which='major', labelsize=font_size*4/5)

        if plot_peaks:
            # max_index = max_values_index[t]
            ax2.plot(max_values_distance[t, :], max_values[t, :], marker = peak_front_marker,color = peak_color,
                     linestyle =' ', markersize=10)
            ax2.plot(front_values_distance[t, :], front_values[t, :], marker = peak_front_marker,color = front_color,
                     linestyle =' ', markersize=10)
            ax2.plot(trail_values_distance[t, :], trail_values[t, :], marker=peak_front_marker, color=front_color,
                     linestyle=' ', markersize=10)


        if plot_wavepeak:
            # max_index = max_values_index[t]
            wave_peak_distance = d_peak_mat[j,t,:][wave_ind_mat_2[j,t,:] != 0]
            wave_peak_height = peak_mat[j,t,:][wave_ind_mat_2[j,t,:] != 0]
            ax2.plot(wave_peak_distance, wave_peak_height , marker = peak_front_marker,color = wave_peak_color,
                     linestyle =' ', markersize=10)

            wave_front_distance = d_front_mat[j,t,:][wave_ind_mat_2[j,t,:] != 0]
            wave_front_height = front_mat[j,t,:][wave_ind_mat_2[j,t,:] != 0]
            ax2.plot(wave_front_distance, wave_front_height, marker=peak_front_marker, color=front_color,
                     linestyle=' ', markersize=10)


        ax2.grid(True)
        ax2.set_title('Origin: Lon. %.0f' % (Flare_coordinates.Tx.to_value()) +u'\u2033'+ ', Lat: %.0f'%( Flare_coordinates.Ty.to_value()) +u'\u2033'+
                     ','+str_direct_width,y = 1.04,size = font_size)


        # Time range in frame index is up to 9999 second, so approximately 2.5 hours
        image_path = os.path.join(path_LVL_0_Results_0_Diagnostics_MI,'movie_frame_%4.4i'%(t_sunpy_sec[t])+'.jpg')
        plt.savefig(image_path,dpi = 300)#,bbox_inches='tight') # The + 100 is to get no problem with missing leading zeros


        from PIL import Image
        im = Image.open(image_path)
        width, height = im.size


        # Setting the points for cropped image
        left = 460
        top =  100
        right =  width- 550
        bottom = height-50

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        im1.save(image_path)

        # Only close plots if they are not shown as quicklook
        if not quicklook:
            plt.close()

    ##############################################
    # Create Movie
    ##############################################
    # Only produces the movie if there is no quicklook
    if not quicklook:
        import moviepy.video.io.ImageSequenceClip
        import glob
        from moviepy.video.io.VideoFileClip import VideoFileClip


        image_files = sorted(glob.glob(path_LVL_0_Results_0_Diagnostics_MI + '/*.jpg'))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=7)
        #clip.write_videofile(path_name[:11] + video_name + '.mp4')




        #########################################################################
        # Saving the plot
        #########################################################################

        if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' plot_timeseries_of_lineplot_and_map : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                        'Only the file_path_dict was used')

        if plot_peaks:
            movie_path = os.path.join(path_LVL_0_Results_0_Diagnostics,'movie_all_peaks'+overplot_info+ filename_appendix)
        else:
            movie_path = os.path.join(path_LVL_0_Results_0,'wave_movie' +overplot_info+ filename_appendix)

        clip.write_videofile(movie_path + '.mp4')

        video_clip = VideoFileClip(movie_path+ '.mp4')

        # Set resolution after rotation
        resolution = (1920, 1080)
        resize_clip = video_clip.resized(resolution)
        # Write the compressed video to the output path
        resize_clip.write_videofile(movie_path+'_compressed' + '.mp4', codec="libx264")

    return


def plot_timeseries_of_map(segment_nr, theta_range, map_series, m_ref_height, intensity_mean, intensity_var, time, distance, linplot_ylim,
                                        wave_peak_kwargs, segment_kwargs, str_direct_width, lower_Tx_lim=0, lower_Ty_lim=0, Tx_Ty_range=0,
                                        Tx_Ty_aspect_ratio = False, Tx_lim=[], Ty_lim =[], plot_wavepeak=True, plot_peaks = True,
                                        plot_sector=True, scale_min = 0.5, scale_max = 1.5,
                                        overplot_map = False,overplot_map_series = [],overplot_kwargs = {},
                                        file_path_dict = [], save_path = [],
                                        filename_appendix = [], distance_marker = True, contrast_mode = False,
                                        quicklook = False,auto_framing = True,
                                        add_overlay_function = False, overlay_function = None, overlay_function_keywords = []):
    """ Creates a Movie for 1 sector with only a map plus overlay

    :param segment_nr:      int, index of sector defined by theta range
    :param theta_range:     1d np.array, see SOLERwave_find_fit_great_arcs
    :param map_series:      sunpy.map.Map series object with all maps
    :param m_base:          reference image as sunpy.map.Map
    :param intensity_mean:  3d np.array, see SOLERwave_find_fit_great_arcs
    :param intensity_var:   3d np.array, see SOLERwave_find_fit_great_arcs
    :param time:            list of strings, see SOLERwave_find_fit_great_arcs
    :param distance:        1d np.array, Distance in length units (e.g. [1e6,2e6] * u.m) of astropy.units
    :param linplot_ylim:    1d np.array, y - limit of the lineplot (left panel)
    :param wave_peak_kwargs: dict, Quantities important for the wave plot
                                    {'d_peak_mat':d_peak_mat,'d_front_mat':d_front_mat,'theta_range':theta_range,
                                    'peak_mat':peak_mat,'front_mat':front_mat,'Flare_coordinates':Flare_coordinates,
                                    'wave_ind_mat_2':wave_value_dict['wave_ind_mat_2']}
    :param segment_kwargs: dict, Quantities important for plotting the segments
                                    {'mask_3':mask_3, 'angles_along_arc_range':angles_along_arc_range}
    :param str_direct_width: sting, part of title string with direction and width of sector of interest
    :param lower_Tx_lim:    float, lower Tx limit in arcseconds
    :param lower_Ty_lim:    float, lower Ty limit in arcseconds
    :param Tx_Ty_range:     float, frame width/height in arcseconds from lower Tx/Ty
    :param Tx_Ty_aspect_ratio: bool, if false (default), than Tx_lim and Ty_lim are calculated form
                                        [Tx_lower, Tx_lower + Tx_Ty_range] and [Ty_lower, Ty_lower + Tx_Ty_range]
    :param Tx_lim:          1d np.array, Tx limit in arcseconds
    :param Ty_lim:          1d np.array, Ty limit in arcseconds
    :param plot_wavepeak:   bool, if true plots the wave peaks
    :param plot_peaks:      bool, if true plots all peaks are plotted (wave peaks are overplotted if also true)
    :param plot_sector:     bool,  if true plots the sector outline
    :param map_keywords:    dict, keyword arguments passed to Map class, default using scale_min and scale_max
    :param scale_max:       float, maximum of scale of the base ratio image
    :param scale_min:       float, minimum of scale of the base ratio image
    :param file_path_dict:      dict, holds default file paths
    :param save_path:           string,custom file path, overruled by file_path_dict if both are given
    :param filename_appendix:   string, custom filename_appendix, overruled by file_path_dict if both are given
    :param distance_marker:     bool, if true plots the distance marker on the sector
    :param contrast_mode:       bool, if true activates contrast mode and outlines the sector plot with a white line
    :return:
    """

    import matplotlib.gridspec as gridspec
    from SOLERwave_find_fit_great_arcs import find_coord_from_angles
    import time as tm

    # Converts distance to Mm
    distance = distance.to_value(u.Mm)

    line_color = 'black'
    front_color= 'green'
    peak_color = 'blue'
    wave_peak_color = 'cyan'
    peak_front_marker ='X' #https://matplotlib.org/stable/api/markers_api.html
    font_size = 24
    wave_peak_front_linewidth = 2
    linewidth_sector = 2


    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Feature Plot : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                    'Only the file_path_dict was used')  # TODO: might ues an actual waring package

    # Set Path for movie images to default or to a custom path
    if len(file_path_dict) != 0:
        path_LVL_0_Results_0_Diagnostics = file_path_dict['path_LVL_0_Results_0_Diagnostics']
        path_LVL_0_Results_0 = file_path_dict['path_LVL_0_Results_0']
        filename_appendix = file_path_dict['filename_appendix']
    else:
        path_LVL_0_Results_0_Diagnostics = save_path
        path_LVL_0_Results_0 = save_path

    # Create a movie image folder based on if the plot is diagnostic or general
    if plot_peaks:
        path_LVL_0_Results_0_Diagnostics_MI = os.path.join(path_LVL_0_Results_0_Diagnostics,'Movie_Images_all_Peaks_only_Map')
    else:
        path_LVL_0_Results_0_Diagnostics_MI = os.path.join(path_LVL_0_Results_0_Diagnostics,'Movie_Images_Kinematics_only_Map')
    os.makedirs(path_LVL_0_Results_0_Diagnostics_MI, exist_ok=True)

    # Remove any files in the folder before creating new ones
    try:
        for file in os.listdir(path_LVL_0_Results_0_Diagnostics_MI):
            file_path = os.path.join(path_LVL_0_Results_0_Diagnostics_MI,file)
            os.remove(file_path)
    except:
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Plotting_tools: plot_timeseries_of_lineplot_and_map: problem with filepath')


    overplot_info=''
    if overplot_map:
        overplot_info = overplot_info + '_map_overplotted'
    if add_overlay_function:
        overplot_info = overplot_info + '_overlaid_function'



    j = segment_nr #Staying concistent with using j for numerating the theta angle

    intensity_slice_mean = intensity_mean[:, j, :]
    intensity_slice_std = np.sqrt(intensity_var[:, j, :])

    map_unit = u.dimensionless_unscaled

    if plot_sector:
        mask_3 = segment_kwargs['mask_3']
        angles_along_arc_range = segment_kwargs['angles_along_arc_range']
        Flare_coordinates = wave_peak_kwargs['Flare_coordinates']

        contours1 = coord_for_sector(Flare_coordinates, theta_range, angles_along_arc_range)

    if plot_wavepeak or plot_peaks:
        d_peak_mat = wave_peak_kwargs['d_peak_mat']
        peak_mat = wave_peak_kwargs['peak_mat']
        d_front_mat = wave_peak_kwargs['d_front_mat']
        front_mat = wave_peak_kwargs['front_mat']
        theta_range = wave_peak_kwargs['theta_range']
        Flare_coordinates = wave_peak_kwargs['Flare_coordinates']
        wave_ind_mat_2 = wave_peak_kwargs['wave_ind_mat_2']

        d_peak_mat_aaa = d_peak_mat*1e6/(m_ref_height.fits_header['rsun_ref']) #Convert to angle along arc
        d_front_mat_aaa = d_front_mat*1e6/(m_ref_height.fits_header['rsun_ref']) #Convert to angle anlong arc

        max_values_distance = d_peak_mat[j,:,:]
        front_values_distance = d_front_mat[j,:,:]
        max_values = peak_mat[j,:,:]
        front_values = front_mat[j,:,:]

    #####################################################
    # Limits for the Map Plot
    #####################################################
    # If no range is given and no Tx_lim, use full width
    if (Tx_Ty_range == 0) & (len(Tx_lim)==0):
        Tx_lim = [m_ref_height.bottom_left_coord.Tx.value, m_ref_height.top_right_coord.Tx.value]
    # If no range is given and no Ty_lim, use full width
    if (Tx_Ty_range == 0) & (len(Ty_lim)==0):
        Ty_lim = [m_ref_height.bottom_left_coord.Ty.value, m_ref_height.top_right_coord.Ty.value]

    if not Tx_Ty_aspect_ratio:
        Tx_lim = [lower_Tx_lim,lower_Tx_lim+Tx_Ty_range]
        Ty_lim = [lower_Ty_lim,lower_Ty_lim+Tx_Ty_range]

    if plot_sector & auto_framing:
        additional_area_in_arcsec = 150
        # The tx,ty values of the sector in arc seconds
        tx_vals = contours1.Tx.value
        ty_vals = contours1.Ty.value

        # Add the Flare coordinates in arcsec to the arrays
        tx_vals = np.append(tx_vals, Flare_coordinates.Tx.value)
        ty_vals = np.append(ty_vals, Flare_coordinates.Ty.value)

        # Set a 100 arcsec Border around area of interest
        Tx_lim_needed = [np.min(tx_vals)-additional_area_in_arcsec,np.max(tx_vals)+additional_area_in_arcsec]
        Tx_lim_diff = np.diff(Tx_lim_needed)[0]

        Ty_lim_needed = [np.min(ty_vals)-additional_area_in_arcsec,np.max(ty_vals)+additional_area_in_arcsec]
        Ty_lim_diff = np.diff(Ty_lim_needed)[0]

        if Tx_lim_diff > Ty_lim_diff:
            # Add Area to Ty_lim to make the aspect ratio a square
            Ty_lim_needed[0] = Ty_lim_needed[0]-((Tx_lim_diff-Ty_lim_diff)/2)
            Ty_lim_needed[1] = Ty_lim_needed[1]+((Tx_lim_diff-Ty_lim_diff)/2)
        elif Tx_lim_diff < Ty_lim_diff:
            # Add Area to Tx_lim to make the aspect ratio a square
            Tx_lim_needed[0] = Tx_lim_needed[0] - ((Ty_lim_diff - Tx_lim_diff) / 2)
            Tx_lim_needed[1] = Tx_lim_needed[1] + ((Ty_lim_diff - Tx_lim_diff) / 2)

        Tx_lim = Tx_lim_needed
        Ty_lim = Ty_lim_needed

    roi_bottom_left = SkyCoord(Tx=Tx_lim[0] * u.arcsec, Ty=Ty_lim[0] * u.arcsec, frame=m_ref_height.coordinate_frame)
    roi_top_right = SkyCoord(Tx=Tx_lim[1] * u.arcsec, Ty=Ty_lim[1] * u.arcsec, frame=m_ref_height.coordinate_frame)

    ylim_low,xlim_low = m_ref_height.wcs.world_to_array_index(roi_bottom_left)
    ylim_high,xlim_high = m_ref_height.wcs.world_to_array_index(roi_top_right)


    #####################################################

    theta_angle = (theta_range[j] + theta_range[j + 1]) / 2

    # In case of a quicklook only 4 images are produced
    if quicklook:
        time_index_range = np.linspace(0,len(time)-1,4,dtype=int)

        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' plot_timeseries_of_lineplot_and_map: Quick Look, only 4 images are produces. Rerun with quicklook == False to create a movie')
    else:
        time_index_range = range(len(time))

    # Formating and creating a progress bar
    # https://builtin.com/software-engineering-perspectives/python-progress-bar:
    from tqdm import tqdm
    import sys
    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    text = now + ' plot_timeseries_of_lineplot_and_map: Frames for Movie :'

    for t in tqdm(time_index_range,desc = text,file=sys.stdout):
        #print(f'Video Frame {t} of {len(time)}')


        #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
        fig = plt.figure(figsize=(10, 9))
        #fig.suptitle(plot_title)
        spec2 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, wspace=0.2, hspace=0.3)

        ##############################################
        # Plotting of the Map
        #############################################

        ax = fig.add_subplot(spec2[0, 0], projection=map_series[t])
        im = map_series[t].plot(axes=ax, cmap='inferno',interpolation = 'none',norm=colors.Normalize(vmin=0.5, vmax=1.5))
        ax.set_xlim([xlim_low, xlim_high])
        ax.set_ylim([ylim_low, ylim_high])
        #map_series[t].draw_limb(axes=ax)
        #ax.set_title(m_base.instrument + '  ' + m_base.wavelength.to_string()[:5] +' $\\text{\AA}$ '+ time [t],size=20)
        ax.set_title(Telescope_Instrument_string(m_ref_height)+' ' + str(np.array(time[t],dtype='datetime64[s]')) + ' UT',size=font_size) # ' \\text{$\AA$}
        plt.colorbar(ax = ax)
        ax.figure.axes[1].tick_params(axis="y", labelsize=font_size * 4/5)

        # Over-plots the map with a custom other map
        if overplot_map == True:
            overplot_map_series[t].plot(axes=ax, **overplot_kwargs)

        m_ref_height.draw_limb(axes=ax,color = 'black')

        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()

        ax.set_xlabel(x_label,size=font_size)
        ax.set_ylabel(y_label,size=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size * 4/5)

        if add_overlay_function:
            overlay_function(ax,t,**overlay_function_keywords)
            overplot_info = overplot_info + '_overlaid_function'

        if distance_marker:
            vec1 = np.linspace(0,1000,11)
            vec1 = vec1[vec1 > distance[0]] # all 100 mM increments below are excepted
            vec1 = vec1[vec1 < distance[-1]] # all 100 mM increment above range are excepted
            aaa_100mM = vec1*1e6/(m_ref_height.fits_header['rsun_ref'])

            # Theta values: -1° + T_j, T_j, T_j+1, T_j+1 + 1°
            theta_angle_range = np.array([theta_range[j],theta_range[j], theta_range[j + 1],theta_range[j + 1]])
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones_like( aaa_100mM))
            theta_mat[0,:] = theta_mat[0,:]  -np.pi/180 * 1/np.sin(aaa_100mM)
            theta_mat[3, :] = theta_mat[3, :] + np.pi / 180 * 1/np.sin(aaa_100mM)
            aaa_mat = np.einsum('i,j->ij', np.ones_like(theta_angle_range), aaa_100mM)

            coords = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat)

            coords_meter = coords * u.meter
            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)
            for i in range(len(vec1)):
                if contrast_mode:
                    ax.plot_coord(coords_sky[:2, i], '-w',linewidth = linewidth_sector*2)
                    ax.plot_coord(coords_sky[-2:, i], '-w',linewidth = linewidth_sector*2)
                ax.plot_coord(coords_sky[:2,i], '-k',linewidth = linewidth_sector)
                ax.plot_coord(coords_sky[-2:, i], '-k',linewidth = linewidth_sector)

        if plot_sector:
            if contrast_mode:
                ax.plot_coord(contours1, color='white',linewidth = linewidth_sector*2)
            ax.plot_coord(contours1, color='black',linewidth = linewidth_sector)


        if plot_peaks:
            #theta_angle_range = theta_angle #(theta_range[:-1] + theta_range[1:]) / 2
            theta_angle_range = np.linspace(theta_range[j],theta_range[j+1], 10)
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones(d_peak_mat_aaa.shape[2]))
            #theta_vec = theta_angle_range* np.ones(d_peak_mat_aaa.shape[2])

            ###############################################################################################
            # Search for the coardinates of the Peak and the Peak + 1 Sigma
            ################################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i,:] = d_peak_mat_aaa[j, t, :]

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            #ax.plot_coord(coords_sky, '--',color = peak_color,linewidth = wave_peak_front_linewidth)
            ax.plot_coord(coords_sky, '-', color='black', linewidth=wave_peak_front_linewidth)
            #################################################################################################
            aaa_mat_front = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_front[i, :] = d_front_mat_aaa[j, t, :]

            coords = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_front)

            coords_meter = coords * u.meter
            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color =front_color,linewidth = wave_peak_front_linewidth)


        if plot_wavepeak:
            d_waves_peak_aaa = d_peak_mat_aaa[j,t,:][wave_ind_mat_2[j,t,:] != 0] #Only peaks part of waves are named
            d_waves_front_aaa = d_front_mat_aaa[j,t,:][wave_ind_mat_2[j,t,:] != 0] #Only peaks part of waves are named

            theta_angle_range = np.linspace(theta_range[j],theta_range[j+1], 10)
            theta_mat = np.einsum('i,j->ij', theta_angle_range, np.ones_like(d_waves_peak_aaa))
            #theta_vec = theta_angle_range* np.ones(d_peak_mat_aaa.shape[2])

            ###############################################################################################
            # Search for the coardinates of the Peak and the Peak + 1 Sigma
            ################################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i,:] = d_waves_peak_aaa

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--',color = wave_peak_color,linewidth = wave_peak_front_linewidth)

            ############################################################################################
            aaa_mat_peak = np.zeros_like(theta_mat)
            for i in range(len(theta_angle_range)):
                aaa_mat_peak[i, :] = d_waves_front_aaa

            coords_meter = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat_peak)

            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky, '--', color=front_color,linewidth = wave_peak_front_linewidth)

        image_path = os.path.join(path_LVL_0_Results_0_Diagnostics_MI,'movie_frame_%3.3i'%(t)+'.png')
        plt.savefig(image_path,dpi = 300)#,bbox_inches='tight') # The + 100 is to get no problem with missing leading zeros


        from PIL import Image
        im = Image.open(image_path)
        width, height = im.size

        # Setting the points for cropped image
        #left = 130
        #top = 100
        #right = width-150
        #bottom = height-40

        # Setting the points for cropped image with colorbar
        left = 0#130
        top = 160+120
        right = width-210-170-280
        bottom = height-100

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        im1.save(image_path)

        # Only close plots if they are not shown as quicklook
        if not quicklook:
            plt.close()

    ##############################################
    # Create Movie
    ##############################################
    # Only produces the movie if there is no quicklook
    if not quicklook:
        import moviepy.video.io.ImageSequenceClip
        import glob

        image_files = sorted(glob.glob(path_LVL_0_Results_0_Diagnostics_MI + '/*.png'))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=7)
        #clip.write_videofile(path_name[:11] + video_name + '.mp4')

        #########################################################################
        # Saving the plot
        #########################################################################

        if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' plot_timeseries_of_lineplot_and_map : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                        'Only the file_path_dict was used')

        if plot_peaks:
            movie_path = os.path.join(path_LVL_0_Results_0_Diagnostics,'movie_all_peaks_only_Map'+overplot_info+ filename_appendix)
        else:
            movie_path = os.path.join(path_LVL_0_Results_0,'wave_movie_only_Map' +overplot_info+ filename_appendix)

        clip.write_videofile(movie_path + '.mp4')
    return




def plot_fit_with_wave_features( time,
                                 d_peak_mat,
                                 delta_d_peak_mat,
                                 d_front_mat,
                                 delta_d_front_mat,
                                 d_trail_mat,
                                 delta_d_trail_mat,
                                 peak_mat,
                                 delta_peak_mat,
                                 distance,
                                 waves_feature_index,
                                 waves_fit,
                                 waves_fit_cov,
                                 peak_tracked,
                                 max_nr_peaks_vec,
                                 nr_of_waves_vec,
                                 wave_ind_mat_2,
                                 waves_time_index,
                                 fit_second_feature,
                                 waves_fit_2,
                                 waves_fit_2_cov,
                                 instr_dir_width_title_string,
                                 segment_length,
                                 v_min_fit = 100,
                                 v_max_fit = 2000,
                                 show_all_points = False,
                                 Plot_Grid = True,
                                 file_path_dict = [],
                                 save_path = [],
                                 filename_appendix = [],
                                 parameter_dict = {},
                                 **kwargs):
    """ Plots the peak associated quantities detected by the peak finding algorithm against time. From top to bottom it
    shows the peak distance with respect to the wave origin in Mm, the front edge distance to the origin in Mm,
    the width of the wave calculated by subtracting front edge from trailing edge in Mm and the peak amplitude in %.

    :param time:        list of strings, see SOLERwave_find_fit_great_arcs
    :param d_peak_mat:          3d np.array, peak or fitted peak, see SoLERwave_find_fit_great_arcs
    :param delta_d_peak_mat:    4d np.array, uncertainty associated with d_peak_mat [lower/upper bound,,,]
    :param d_front_mat:         3d np.array, see SoLERwave_find_fit_great_arcs
    :param delta_d_front_mat:   4d np.array, uncertainty associated with d_front_mat [lower/upper bound,,,]
    :param d_trail_mat:         3d np.array, see SoLERwave_find_fit_great_arcs
    :param delta_d_trail_mat:   4d np.array, uncertainty associated with d_trail_mat [lower/upper bound,,,]
    :param peak_mat:            3d np.array, peak or fitted peak, see SoLERwave_find_fit_great_arcs
    :param delta_peak_mat:      3d np.array, uncertainty associated with peak_mat [lower/upper bound,,,]
    :param distance:    1d np.array, Distance in length units (e.g. [1e6,2e6] * u.m) of astropy.units
    :param waves_feature_index: 1d np.array in list of lists, see SoLERwave_find_fit_great_arcs
    :param waves_fit:           1d np.array in list of lists, see SoLERwave_find_fit_great_arcs
    :param waves_fit_cov:       1d np.array in list of lists, see SoLERwave_find_fit_great_arcs
    :param peak_tracked:        bool, Adds "peak" or "front" to the title
    :param max_nr_peaks_vec:    1d np.array, see SoLERwave_find_fit_great_arcs
    :param nr_of_waves_vec:     1d np.array, see SoLERwave_find_fit_great_arcs
    :param wave_ind_mat_2:      3d np.array, see SoLERwave_find_fit_great_arcs
    :param waves_time_index:    1d np.array in list of lists, see SoLERwave_find_fit_great_arcs
    :param fit_second_feature:  bool, plost also the fit for a secondary feature if true
    :param waves_fit_2:         1d np.array in list of lists, required for secondary feature
    :param waves_fit_2_cov:     1d np.array in list of lists, required for secondary feature
    :param instr_dir_width_title_string: string, part of the title of the plot, see tutorial jupyter notebook
                                                 of SOLERwave tool for more information
    :param segment_length:  float with astropy lenght units, length of a segment
    :param v_min_fit:       float, lower limit for the fitted wave velocities that are shown, default 100 km/s
    :param v_max_fit:       float, upper limit for the fitted wave velocities that are shown, default 2000 km/s
    :param show_all_points: bool, if true, all point are plotted
    :param Plot_Grid:       bool, if true, a grid is plotted
    :param file_path_dict:      dict, holds default file paths
    :param save_path:           string,custom file path, overruled by file_path_dict if both are given
    :param filename_appendix:   string, custom filename_appendix, overruled by file_path_dict if both are given
    :param parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    :param kwargs: catches all additional keyword arguments, allowing the function to be cast by a dict with to many
                    entries
    :return:
    """


    from sunpy.time import parse_time
    import sunpy.timeseries

    distance_MM = distance.to_value('Mm')

    marker_wave = ['x', '+', '1','2']
    #marker_wave = ['v','^','<','>']
    marker_vec_all_peaks = ['s','o','^','v','X']
    marker = ['x']
    #colors = {
    #    'blue': '#377eb8',
    #    'orange': '#ff7f00',
    #    'green': '#4daf4a',
    #    'pink': '#f781bf',
    #    'brown': '#a65628',
    #    'purple': '#984ea3',
    #    'gray': '#999999',
    #    'red': '#e41a1c',
    #    'yellow': '#dede00'
    #}
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    #https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
    CB_color_cycle = ['#D81B60','#1E88E5','#FFC107','#004D40']

    label_size = 12

    v_min_fit_Mm = v_min_fit / 1e3  # converts form km/s to Mm/s
    v_max_fit_Mm = v_max_fit / 1e3  # converts form km/s to Mm/s


    time_dateobj = np.array(time, dtype='datetime64[ns]')

    time_sunpyobj = parse_time(time)
    t_sunpy_sec = (time_sunpyobj - time_sunpyobj[0]).to_value('sec')

    if peak_tracked:
        Title_Tracked_feature = 'Peak'
    else:
        Title_Tracked_feature = 'Front'



    for j in range(d_front_mat.shape[0]):


        fig, ax = plt.subplots(4, figsize=(10, 10),
                               gridspec_kw={'wspace': 0.2, 'hspace': 0.1},sharex=True)  # figsize=(15, 10),

        if show_all_points:
            show_all_text = ' Show all Points'
        else:
            show_all_text = ''

        plot_title_kinematics = ('Wave Tracking: ' + Title_Tracked_feature + show_all_text + '\n'
                          + instr_dir_width_title_string)
        fig.suptitle(plot_title_kinematics,y=0.95)

        # Name axis for easier use later
        ax_d_peak = ax[0]
        ax_d_front = ax[1]
        ax_width = ax[2]
        ax_height = ax[3]

        x_time_lim = [len(time_dateobj),0]

        # Jump over most of the plotting routines if no peak was found
        if max_nr_peaks_vec[j] != 0:

            # Plot all points from their corresponding matrices
            if show_all_points:
                for t_index in range(len(time_dateobj)):
                    for mu_index in range(d_peak_mat.shape[2]):
                        color_plot = CB_color_cycle[mu_index % len(CB_color_cycle)]
                        marker_all_peaks = marker_vec_all_peaks[mu_index // len(CB_color_cycle)]
                        ax_d_peak.plot(time_dateobj[t_index],
                                        d_peak_mat[j, t_index, mu_index],
                                        marker_all_peaks,
                                        color=color_plot)
                        ax_d_front.plot(time_dateobj[t_index],
                                        d_front_mat[j, t_index, mu_index],
                                        marker_all_peaks,
                                        color=color_plot)
                        ax_width.plot(time_dateobj[t_index],
                                      d_front_mat[j, t_index, mu_index] - d_trail_mat[j, t_index, mu_index],
                                      marker_all_peaks,
                                      color=color_plot)
                        ax_height.plot(time_dateobj[t_index],
                                        peak_mat[j, t_index, mu_index],
                                        marker_all_peaks,
                                        color=color_plot)
            # Iterate over all the total nr of waves in this theta segment j
            for nr_w in range(nr_of_waves_vec[j]):

                # Creating vectors corresponding to the entries of one identified wave using the
                # index of individual features
                ind_vec = waves_feature_index[j][nr_w]
                wave_d_peak_vec  = d_peak_mat[j].flatten()[ind_vec]
                wave_d_front_vec = d_front_mat[j].flatten()[ind_vec]
                wave_d_trail_vec = d_trail_mat[j].flatten()[ind_vec]
                wave_peak_vec    = peak_mat[j].flatten()[ind_vec]

                delta_wave_d_peak_vec  = np.array([delta_d_peak_mat[0,:,:,:][j].flatten()[ind_vec],
                                                   delta_d_peak_mat[1,:,:,:][j].flatten()[ind_vec]])
                delta_wave_peak_vec    = delta_peak_mat[j].flatten()[ind_vec]
                delta_wave_d_front_vec = np.array([delta_d_front_mat[0,:,:,:][j].flatten()[ind_vec],
                                                   delta_d_front_mat[1,:,:,:][j].flatten()[ind_vec]])
                delta_wave_d_trail_vec = np.array([delta_d_trail_mat[0,:,:,:][j].flatten()[ind_vec],
                                                   delta_d_trail_mat[1,:,:,:][j].flatten()[ind_vec]])

                #delta_w_d_f_v = delta_d_front_mat[j].flatten()[ind_vec]
                #delta_wave_d_front_vec = np.zeros((2,len(delta_w_d_f_v)))
                #delta_wave_d_front_vec[0,:] = delta_w_d_f_v + segment_uncertainty #Segement uncertainty trails the front
                #elta_wave_d_front_vec[1,:] = delta_w_d_f_v



                # Fitting parameters gained from wave tracing algorithm
                a = waves_fit[j][nr_w]
                fit_cov = waves_fit_cov[j][nr_w]

                # Fitting parameters for secondary feature gained from wave tracing algorithm
                if fit_second_feature:
                    a_2 = waves_fit_2[j][nr_w]
                    fit_cov_2 = waves_fit_2_cov[j][nr_w]

                # Create a numpy vector from the timestamps index corresponding to the wave feature index
                time_index_vec = np.array(waves_time_index[j][nr_w])

                # Checks of the fitted wave is within the speed range
                if (a[0] >= v_min_fit_Mm) and (a[0] <= v_max_fit_Mm):
                    t_vec = np.array([time_index_vec[0], time_index_vec[-1]])
                    fit_vec = np.poly1d(a)(t_sunpy_sec[t_vec])

                    if not show_all_points:
                        # Iterates through colors for each wave
                        color_plot = CB_color_cycle[nr_w % len(CB_color_cycle)]

                        # Plot all points for each fitted wave if not all peaks are plotted
                        #plt.errorbar(time_dateobj[time_vec], wave_p_vec, std_vec, None, 'x', color='black')
                        wave_width = (wave_d_front_vec - wave_d_trail_vec)
                        #delta_wave_width = np.sqrt(delta_wave_d_trail_vec**2+delta_wave_d_front_vec**2)
                        delta_wave_width = np.zeros_like(delta_wave_d_front_vec) *np.nan
                        delta_wave_width[0,:] = delta_wave_d_trail_vec[1,:] + delta_wave_d_front_vec[0,:] # Largest Error Methode
                        delta_wave_width[1, :] = delta_wave_d_trail_vec[0, :] + delta_wave_d_front_vec[1, :] # Largest Error Methode

                        # Calculates the minimum wave width including uncertainties
                        min_wave_width = wave_width - delta_wave_width[0,:]

                        # Check if the min_wave_width is smaller than half the segment width
                        to_small_wavewidth = (min_wave_width<(segment_length/2))

                        # Set all delta_wave_width so that delta_wave_width + segment_length/2 = wave_width
                        delta_wave_width[0,to_small_wavewidth] = wave_width[to_small_wavewidth]-segment_length/2

                        # In the case the wave_width itself being smaller than segment/2,
                        # lower bound for uncertainty is set to 0
                        delta_wave_width = np.where(delta_wave_width<0,0,delta_wave_width)

                        ax_d_peak.errorbar(time_dateobj[time_index_vec],
                                           wave_d_peak_vec,
                                           yerr = delta_wave_d_peak_vec,
                                           marker = marker[nr_w % len(marker)],
                                           color=color_plot,
                                           linestyle = '')
                        ax_d_front.errorbar(time_dateobj[time_index_vec],
                                            wave_d_front_vec,
                                            yerr = delta_wave_d_front_vec,
                                            marker =marker[nr_w % len(marker)],
                                            color=color_plot,
                                            linestyle = '')
                        ax_width.errorbar(time_dateobj[time_index_vec],
                                          wave_width,
                                          yerr = delta_wave_width,
                                          marker = marker[nr_w % len(marker)],
                                          color=color_plot,
                                          linestyle = '')
                        ax_height.errorbar(time_dateobj[time_index_vec],
                                           wave_peak_vec,
                                           yerr = delta_wave_peak_vec,
                                           marker =marker[nr_w % len(marker)],
                                           color=color_plot,
                                           linestyle = '')

                        if min(time_index_vec) < x_time_lim[0]:
                            x_time_lim[0] = min(time_index_vec)
                        if max(time_index_vec) > x_time_lim[1]:
                            x_time_lim[1] = max(time_index_vec)

                    if show_all_points:
                        color_plot = 'black'
                        marker_w = marker_wave[nr_w % len(marker_wave)]
                        marker_size = 10
                    else:
                        color_plot = CB_color_cycle[nr_w % len(CB_color_cycle)]
                        marker_w = 'None'
                        marker_size = 5

                    if peak_tracked:
                        ax_d_peak.plot(time_dateobj[t_vec], fit_vec, '-',color = color_plot,marker=marker_w,markersize = marker_size,
                                label='v = %.f $\pm$ %.f km/s ' % (
                                a[0] * 1000, np.sqrt(fit_cov[0]) * 1000))
                        ax_d_peak.legend()

                        if fit_second_feature:
                            fit_vec = np.poly1d(a_2)(t_sunpy_sec[t_vec])
                            ax_d_front.plot(time_dateobj[t_vec], fit_vec, '-', color=color_plot,marker=marker_w,markersize = marker_size,
                                            label='v = %.f $\pm$ %.f km/s ' % (
                                                a_2[0] * 1000, np.sqrt(fit_cov_2[0]) * 1000))
                            ax_d_front.legend()
                    else:



                        ax_d_front.plot(time_dateobj[t_vec], fit_vec, '-',color = color_plot,marker=marker_w,markersize = marker_size,
                                       label='v = %.f $\pm$ %.f km/s ' % (
                                           a[0] * 1000, np.sqrt(fit_cov[0]) * 1000))
                        ax_d_front.legend()

                        if fit_second_feature:
                            fit_vec = np.poly1d(a_2)(t_sunpy_sec[t_vec])
                            ax_d_peak.plot(time_dateobj[t_vec], fit_vec, '-', color=color_plot,marker=marker_w,markersize = marker_size,
                                           label='v = %.f $\pm$ %.f km/s ' % (
                                               a_2[0] * 1000, np.sqrt(fit_cov_2[0]) * 1000))
                            ax_d_peak.legend()


            if (not show_all_points) and (nr_of_waves_vec[j] == 0):
                # Adds a text in the case of no wave found
                ax_d_peak.set_ylim([distance_MM[0], distance_MM[-1] * 1.2])
                ax_d_front.set_ylim([distance_MM[0], distance_MM[-1] * 1.2])
                # Adds +-2 min just to have space in case of few points
                ax_height.set_xlim(time_dateobj[0] - np.timedelta64(2, 'm'),
                                   time_dateobj[-1] + np.timedelta64(2, 'm'))

                pos_index = int(len(time_dateobj) * 1/3 )

                ax_d_peak.text(time_dateobj[pos_index], distance_MM[-1] * 0.6, 'no wave found', color='black',
                               fontweight='bold')

            elif not show_all_points:
                # Set the xlim of the time axis to -2 min before the first and +2 min after the last wave feature
                ax_height.set_xlim(time_dateobj[x_time_lim[0]] - np.timedelta64(2, 'm'),
                                   time_dateobj[x_time_lim[1]] + np.timedelta64(2, 'm'))

            else:
                # In the case of show_all_points=True, +-2 min to full time line
                ax_height.set_xlim(time_dateobj[0] - np.timedelta64(2, 'm'),
                                   time_dateobj[-1] + np.timedelta64(2, 'm'))

            d_y_lim= [ax_d_peak.get_ylim()[0],ax_d_front.get_ylim()[1]]

            #ax_d_peak.set_title('Peak distance',x=-0.1,y=0.5) # rotation='vertical',x=-0.1,y=0.5
            ax_d_peak.set_ylabel('Peak Distance (Mm)',size = label_size)
            ax_d_peak.tick_params(axis='y', which='major', labelsize=label_size)
            ax_d_peak.grid(Plot_Grid)
            ax_d_peak.set_ylim(d_y_lim)

            #ax_d_front.set_title('Front distance',x=-0.1,y=0.5)
            ax_d_front.set_ylabel('Front Distance (Mm)',size = label_size)
            ax_d_front.tick_params(axis='y', which='major', labelsize=label_size)
            ax_d_front.grid(Plot_Grid)
            ax_d_front.set_ylim(d_y_lim)

            #ax_width.set_title('Wave width',x=-0.1,y=0.5)
            ax_width.set_ylabel('Wave Width (Mm)',size = label_size)
            ax_width.tick_params(axis='y', which='major', labelsize=label_size)
            ax_width.grid(Plot_Grid)
            #ax_height.set_title('Peak height',x=-0.1,y=0.5)
            ax_height.set_ylabel('Peak Amplitude',size = label_size)
            ax_height.tick_params(axis='y', which='major', labelsize=label_size)
            ax_height.grid(Plot_Grid)
            #ax_height.set_ylim([1,1.15])# [1,1.06])
            #ax_height.plot(time_dateobj,np.ones(len(time)),'--',color = 'red')
            #ax_height.locator_params(tight=True, nbins=8)


            # https://matplotlib.org/1.5.3/examples/pylab_examples/date_demo1.html
            # Used below
            # https://matplotlib.org/stable/gallery/ticks/date_concise_formatter.html
            import matplotlib.dates as mdates
            from matplotlib.ticker import MultipleLocator, AutoMinorLocator
            locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            ax_height.xaxis.set_major_locator(locator)
            ax_height.xaxis.set_major_formatter(formatter)

            # https://matplotlib.org/3.4.3/gallery/ticks_and_spines/major_minor_demo.html
            ax_height.xaxis.set_minor_locator(AutoMinorLocator(4))

            ax_height.tick_params(axis='x', which='major', labelsize=label_size)
            ax_height.xaxis.get_offset_text().set_size(label_size)

        else:
            #ax_d_peak.set_title('Direction %.f degree ' % (theta_angle * 180 / np.pi))
            ax_d_peak.set_ylim([distance_MM[0], distance_MM[-1] * 1.2])
            ax_d_peak.set_xlim([time_dateobj[0], time_dateobj[-1]])
            # transparent_white = (1, 1, 1, 0.5)
            # Sets the text at the datetimeobject roughtly 1/3 in the timeline
            pos_index = int(len(time_dateobj) * 1 / 3)
            ax_d_peak.text(time_dateobj[pos_index], distance_MM[-1] * 0.6, 'no peak found', color='black',
                        fontweight='bold')

    #########################################################################
    # Saving the plot
    #########################################################################

    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Feature Plot : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                    'Only the file_path_dict was used')

    if len(file_path_dict) != 0:
        if show_all_points:
            save_path = file_path_dict['path_LVL_0_Results_0_Diagnostics']
            filename_appendix = file_path_dict['filename_appendix']
        else:
            save_path = file_path_dict['path_LVL_0_Results_0']
            filename_appendix = file_path_dict['filename_appendix']

    if len(filename_appendix) == 0:
        filename_appendix = ''

    if len(save_path) != 0:
        if show_all_points:
            feature_path = os.path.join(save_path, 'Kinematics_all_Peaks' + filename_appendix+'.png')
            feature_path_pdf = os.path.join(save_path, 'Kinematics_all_Peaks' + filename_appendix + '.pdf')
        else:
            feature_path = os.path.join(save_path, 'Kinematics' + filename_appendix + '.png')
            feature_path_pdf = os.path.join(save_path, 'Kinematics_all_Peaks' + filename_appendix + '.pdf')
        plt.savefig(feature_path,dpi = 300,format = 'png')
        plt.savefig(feature_path_pdf,dpi = 300, format='pdf')


        '''
        from PyPDF2 import PdfWriter, PdfReader, PdfMerger

        reader = PdfReader(feature_path)
        writer = PdfWriter()

        page = reader.pages[0]
        print(page.cropbox.lower_left)
        print(page.cropbox.lower_right)
        print(page.cropbox.upper_left)
        print(page.cropbox.upper_right)

        left = 50  # 130
        top = 40
        right = page.cropbox.upper_right[1] - 80
        bottom =page.cropbox.lower_right[0]- 50

        page.mediabox.lower_right = (right, bottom)
        page.mediabox.lower_left = (left, bottom)
        page.mediabox.upper_right = (right, top)
        page.mediabox.upper_left = (left, top)
        '''

        #writer.add_page(page)

        #with open(feature_path, 'wb') as fp:
        #    writer.write(fp)


        from PIL import Image
        im = Image.open(feature_path)
        width, height = im.size


        # Setting the points for cropped image
        left = 50  # 130
        top = 40
        right = width - 80
        bottom = height - 50

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        im1.save(feature_path)

    parameter_dict['plot_fit_with_wave_features:'] = ' '
    parameter_dict['plot_fit_with_wave_features: show_all_points = '+ str(show_all_points)] = ' '
    parameter_dict['plot_fit_with_wave_features: show_all_points = '+ str(show_all_points) + ', v_min_fit (km/s)'] = v_min_fit
    parameter_dict['plot_fit_with_wave_features: show_all_points = ' + str(show_all_points) + ', v_max_fit (km/s)'] = v_max_fit


    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + ' Plotting_tools: fit_waves_with_subplots finished')
    return parameter_dict