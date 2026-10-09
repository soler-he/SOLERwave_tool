import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric
import numpy as np
from numba import njit, prange
from scipy.signal import find_peaks
import time as tm


def pixel_to_great_segments(coord_of_flare,
                            m_0,
                            r_sun_pixel,
                            flat_field = False):
    ''' Assigns an azimuthal and radial angle to each pixel of a map, starting from an origin

    :param coord_of_flare:  SkyCoord object, center of the flare with coord_frame of base map
    :param m_0:             map object with wcs, defines the wcs of the calculation; normally first map in the sequence to investigate
    :paraM r_sun_pixel      float, radius of the sun in pixel
    :param flat_field:      bool, switches between flat field at sun distance and following the sun curvature
    :return:
        pppixel_vectors_xyz: 3d np.array, carthesian coordinates of each pixel in [xyz,pixel_i,pixel_j] == [xyz,pixel_y,pixel_x]
        aangles_along_arc:   2d np.array, angles of concetric circles around the flare for each pixel
        ttheta:              2d np.array, mathematicaly positve angles starting from the north intersecting great arc for each pixel

    ###########################################
    Example how to use this function:

    Flare_coordinates = SkyCoord(Tx=292 * u.arcsec, Ty=127 * u.arcsec, frame=m_seq_base.maps[0].coordinate_frame)
    coords,aangles_along_arc,ttheta = pixel_to_great_segments(Flare_coordinates,m_seq[0])
    ###########################################
    '''
    import sunpy.sun.constants

    py,px = m_0.data.shape #Note: This line was added in the submit 2024.11.11, older programms might require adaptiations
    px_range = np.arange(px)
    py_range = np.arange(py)

    if not(flat_field):
        ########## Handle inputs for only

        start = coord_of_flare.transform_to(Heliocentric)

        distance_unit = u.m #start.cartesian.xyz.unit

        center = SkyCoord(0 * distance_unit,
                          0 * distance_unit,
                          0 * distance_unit,
                          obstime=start.obstime,
                          observer=start.observer,
                          frame=Heliocentric)

        start_cartesian = start.cartesian.xyz.to(distance_unit).value
        # end_cartesian = end.cartesian.xyz.to(start.distance_unit).value
        center_cartesian = center.cartesian.xyz.to(distance_unit).value

        v1 = start_cartesian - center_cartesian

        r_sun = np.linalg.norm(v1)

        # Defines the second Vector to point to the north pole
        # Form Heliocentric coordinates:
        # The Y-axis is aligned with the component of the vector to the Sun’s north pole that is perpendicular to the Z-axis.
        v2 = np.array([0, r_sun, 0])

        #Vector in the plane of v1,v2 perp to v1
        v3 = np.cross(np.cross(v1, v2), v1)
        v3 = r_sun * v3 / np.linalg.norm(v3)

        #Vector pero to v1 and perp to v3
        v4 = np.cross(v1, v3) / r_sun


        ppx, ppy = np.meshgrid(px_range, py_range)

        ppixel_vectors = m_0.wcs.array_index_to_world(ppy,ppx)      #From astropy doc: (i, j) order, where for an image i is the row and j is the column
        ppixel_vectors = ppixel_vectors.transform_to(Heliocentric)

        pppixel_vectors_xyz = ppixel_vectors.cartesian.xyz.value


        aangles_along_arc = np.arctan2(np.linalg.norm(np.cross(v1, pppixel_vectors_xyz, axisb=0,axisc=0),axis=0),np.einsum('i,ijk->jk',v1, pppixel_vectors_xyz))

        ttheta = np.arctan2( np.einsum('i,ijk->jk',v4, pppixel_vectors_xyz),np.einsum('i,ijk->jk',v3, pppixel_vectors_xyz))

    else:
        ############################
        # Flat field calculation
        ###########################
        r_sun = sunpy.sun.constants.equatorial_radius.to('Mm')  # Todo: Solar Radius

        l_pixel = r_sun /r_sun_pixel

        x_flare,y_flare = m_0.wcs.world_to_pixel(coord_of_flare)
        # Find the pixel angles
        x_pixel, y_pixel = np.meshgrid(px_range - x_flare,
                                       py_range - y_flare)

        ttheta = np.arctan2(y_pixel,x_pixel)-np.pi/2#*180/np.pi
        ppixel_distance = np.sqrt(x_pixel**2+y_pixel**2)

        aangles_along_arc = ppixel_distance*l_pixel.to_value('Mm')
        pppixel_vectors_xyz = None

    return pppixel_vectors_xyz,aangles_along_arc,ttheta

def find_segments_from_list(theta_range, ttheta, angles_along_arc_range, aangles_along_arc, m_data_list):
    '''Finds the segments between the theta_range and angles_along_arc_range vectors and produces a mask and mean/var

    :param theta_range: 1d np.array, vector of the segment borders along ttheta direction
    :param ttheta:      2d np.array, mathematically positive angles starting from the north intersecting great arc for each pixel, return of pixel_to_great_segments
    :param angles_along_arc_range: 1d np.array, vector of the segment borders along the arc direction
    :param aangles_along_arc:      2d np.array, angles of concentric circles around the flare for each pixel, return of pixel_to_great_segments
    :param m_data_list:      list of 2d np.array, maps to be investigated [time][py,px]
    :return:
        intensity_mean  3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
        intensity_median_staggered  3d np.array, median of the intensity of each segment [angles_along_arc, theta, time]
        intensity_var   3d np.array, variance of the intensity of each segment [angles_along_arc, theta, time]
        pixel_per_segment 3d np.array, pixel per segment (same for all timesteps) [angles_along_arc, theta, time]
        mask_3          2d np.array, same size as ppx/ppy, mask of the different segments, numeration starting with 1, than following arcs, than theta

    ###########################################
    Example how to use this function (Note, not all required inputs are listed)

    map_data_list,m_base,time,m_seq_base,... = load_preprocessed_fits(path =path,LVL_0_directory = LVL_0_directory,...)

    coords,aangles_along_arc,ttheta = pixel_to_great_segments(...)
    intensity_mean,intensity_var,mask_3 = find_segments(theta_range,ttheta,angles_along_arc_range,aangles_along_arc,m_data_list)
    #############################################
    '''


    py, px = m_data_list[0][:, :].data.shape  # Note: This line was added in the submit 2024.11.11, older programms might require adaptiations
    px_range = np.arange(px)
    py_range = np.arange(py)

    ppx, ppy = np.meshgrid(px_range, py_range)

    from numba import jit, njit, prange
    # Find ranges to allow for faster processing later
    # the range is one shorter as the vector as the last value marks the outer boundary
    i_range = angles_along_arc_range.shape[0] - 1
    j_range = theta_range.shape[0] - 1
    k_range = aangles_along_arc.shape[0]
    l_range = aangles_along_arc.shape[1]

    mask_3 = np.zeros_like(ttheta)

    intensity_mean = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list))) *np.nan
    intensity_var = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list)))  *np.nan
    pixel_per_segment = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list))) *np.nan
    intensity_median = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list)))  *np.nan

    # Shifting all Theta Values to positive values
    # Segment bound assures the change in index is at a boundary
    segment_bound = theta_range[0]
    # ttheta[ttheta < segment_bound] = ttheta[ttheta < segment_bound] + 2 * np.pi - segment_bound
    # theta_range[theta_range <= segment_bound] = theta_range[theta_range <= segment_bound] + 2 * np.pi - segment_bound

    ttheta = ttheta - segment_bound
    ttheta[ttheta < 0] = ttheta[ttheta < 0] + 2 * np.pi
    theta_range = theta_range - segment_bound
    theta_range[theta_range < 0] = theta_range[theta_range < 0] + 2 * np.pi


    # njited paralell loop to find the values
    for_loop_for_list(theta_range, ttheta, angles_along_arc_range, aangles_along_arc, ppx, ppy, intensity_mean, intensity_var,intensity_median,
               pixel_per_segment, m_data_list, mask_3, i_range, j_range, k_range, l_range)




    return intensity_mean, intensity_median,intensity_var, pixel_per_segment, mask_3

@njit(parallel=True)
def for_loop_for_list(theta_range,ttheta,angles_along_arc_range,aangles_along_arc,ppx,ppy,intensity_mean,intensity_var,intensity_median,pixel_per_segment,m_data_list,mask_3,i_range,j_range,k_range,l_range):
    """Function to be called in find segments to speed up the process

    :param defined in the find_segments_from_list
    :return Cast to the input matrices
    """
    time_sequ_len = len(m_data_list)

    for i in prange(i_range):       # angle_along_arc
        for j in range(j_range):    # theta


            #data_segement = [[1] for _ in range(m_data.shape[2])]
            data_mean = np.zeros(time_sequ_len)
            data_mean_square = np.zeros(time_sequ_len)
            data_segment_count = np.zeros(time_sequ_len,dtype=np.int32)
            data_median = np.zeros((time_sequ_len,100000))

            #test  = np.where(angles_along_arc_range[i] < aangles_along_arc < angles_along_arc_range[i + 1],True,False)
            # 0.08 sec for px_mask, py_mask and mask_2

            # theta [x,y] = theta [k,l]
            for k in range(k_range):
                for l in range(l_range):
                    #mask[k, l] =
                    if bool((angles_along_arc_range[i] <= aangles_along_arc[k, l] < angles_along_arc_range[i + 1]) & (theta_range[j] <= ttheta[k, l] < theta_range[j + 1])):  # np.bool_(mask_i[:,:,i])
                        px = ppx[k,l]
                        py = ppy[k,l]


                        for index in prange(time_sequ_len):
                            map_data_point = m_data_list[index][py,px]

                            if ~np.isnan(map_data_point):
                                data_mean[index] += map_data_point
                                data_mean_square[index] += map_data_point**2
                                data_median[index,data_segment_count[index]] = map_data_point

                                data_segment_count[index] +=1

                        mask_3[k,l] = j*(angles_along_arc_range.shape[0]-1) + i + 1


            for index in prange(time_sequ_len):
                # Checks if there was any not nan value in the
                if data_segment_count[index] != 0:
                    intensity_mean[i, j, index] = data_mean[index]/data_segment_count[index]
                    intensity_var[i, j, index] = data_mean_square[index]/data_segment_count[index]-(data_mean[index]/data_segment_count[index])**2
                    pixel_per_segment[i, j,index] = data_segment_count[index]

                    intensity_median[i,j,index] = np.median(data_median[index,:data_segment_count[index]])

def find_segments_from_list_staggered(theta_range,
                                      ttheta,
                                      angles_along_arc_range,
                                      aangles_along_arc,
                                      map_data_list,
                                      r_sun_ref,
                                      r_sun_pixel,
                                      m_ref_height,
                                      times_staggered = 4,
                                      calculate_uncertainty = True,
                                      flat_field = False,
                                      parameter_dict ={}):
    """ Wrapper to call the find_segments_from_list function multiple times for a staggered output for the segments
    between the theta_range and angles_along_arc_range. Staggering is done equidistant along the angles_along_arc
    direction.

    :param theta_range: 1d np.array, vector of the segment borders along ttheta direction
    :param ttheta:      2d np.array, mathematically positive angles starting from the north intersecting great arc for each pixel, return of pixel_to_great_segments
    :param angles_along_arc_range: 1d np.array, vector of the segment borders along the arc direction
    :param aangles_along_arc:      2d np.array, angles of concentric circles around the flare for each pixel, return of pixel_to_great_segments
    :param m_data_list:            list of 2d np.array, maps to be investigated [time][py,px]
    :param r_sun_ref:              astropy unit float (in meter), reference sphere the calculations take place on
    :param r_sun_pixel:            float, radius of the sun in pixel
    :param times_staggered:        int, number of steps between two angles in angle_along_arc_range including the first
    :param calculate_uncertainty:   bool, activate tracking of the segment uncertainty depending on segment position
    :param flat_field:              bool, switches between flat field at sun distance and following the sun curvature
    :param parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        intensity_mean_staggered  3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
        intensity_median_staggered  3d np.array, median of the intensity of each segment [angles_along_arc, theta, time]
        intensity_var_staggered   3d np.array, variance of the intensity of each segment [angles_along_arc, theta, time]
        distance_staggered        1d np.array, distance values in u.meter
        base_mask                 2d np.array, same size as ppx/ppy, mask of the different segments of the non-staggered
                                               map, numeration starting with 1, than following arcs, than theta
        base_pixel_per_segment    3d np.array, pixel per segment for non-staggered map [angles_along_arc, theta, time]
                                               same value for all time steps
        parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions
    """

    assert aangles_along_arc.shape == ttheta.shape
    assert aangles_along_arc.shape == map_data_list[0].shape

    def f_median(mat, **kwargs):
        '''
        :param mat:     2d np.array, [data,time]
        :param kwargs:  unused
        :return:        1d np.array, median of the data [time]

        Call:
        intensity_median = evaluate_on_mask_from_list(base_mask,f_median,map_data_list,theta_range,angles_along_arc_range,base_pixel_per_segment,func_dimensions = 1)
        '''
        return np.expand_dims(np.median(mat, axis=0),1)

    def f_max_pixel_distance(mat,**kwargs):
        '''
        :param mat:     2d np.array, [data,time]
        :param kwargs:  unused
        :return:        1d np.array, max pixel distance to center in segment [time]
        '''
        if mat.size != 0:
            a = np.expand_dims(np.nanmax(mat, axis=0), 1)
        else:
            #print('asdf')
            a = np.nan

        return a


    import sunpy
    import astropy.units as u

    t1 = tm.time()


    intensity_mean,intensity_median, intensity_var, base_pixel_per_segment, base_mask = find_segments_from_list(theta_range, ttheta,
                                                                                       angles_along_arc_range,
                                                                                       aangles_along_arc, map_data_list)

    ts = times_staggered


    ################### Staggered Plot (with 3 steps between each angle along arch ) ##############################
    #'''
    ##########################################################################################
    diff_angles = (angles_along_arc_range[1] - angles_along_arc_range[0]) / ts

    if flat_field:
        distance = (angles_along_arc_range[:-1] + np.diff(angles_along_arc_range) / 2)*1e6*u.m
        diff_distance = diff_angles * 1e6*u.m
    else:
        distance = (angles_along_arc_range[:-1]+np.diff(angles_along_arc_range)/2) * r_sun_ref
        diff_distance = diff_angles * r_sun_ref


    intensity_mean_staggered = np.zeros((intensity_mean.shape[0]*ts,intensity_mean.shape[1],intensity_mean.shape[2]))
    intensity_mean_staggered[::ts,:,:] = intensity_mean[:,:,:]

    intensity_median_staggered = np.zeros_like(intensity_mean_staggered)
    intensity_median_staggered[::ts, :, :] = intensity_median[:, :, :]

    intensity_var_staggered = np.zeros_like(intensity_mean_staggered)
    intensity_var_staggered[::ts,:,:] = intensity_var[:,:,:]/base_pixel_per_segment # The variance is given for the peak intensity: var_mean = var/N

    distance_staggered = np.zeros(((len(angles_along_arc_range)-1)*ts))
    distance_staggered[::ts] = distance

    max_pixel_per_segment = [np.max(base_pixel_per_segment)]
    min_pixel_per_segment = [np.min(base_pixel_per_segment)]

    ########Pixel Uncertainty #####################
    #
    ###############################################
    #'''
    if calculate_uncertainty and (not flat_field):
        # Coordinates Flares
        Flare_coordinates = SkyCoord(Tx = 0* u.arcsec, Ty = 0* u.arcsec,frame=m_ref_height.coordinate_frame)

        # Find the pixel angles
        _, aaa_central, ttheta_central = pixel_to_great_segments(Flare_coordinates, m_ref_height,r_sun_pixel)


        #CRPIX1  [pixel] CRPIX1: location of sun center in CCD x
        #CRPIX2 [pixel] CRPIX2: location of sun center in CCD y
        x_pixel,y_pixel = np.meshgrid(np.arange(ttheta_central.shape[0])-m_ref_height.fits_header['CRPIX1'],
                                      np.arange(ttheta_central.shape[1])-m_ref_height.fits_header['CRPIX2'])

        x_pixel_center = np.sqrt(x_pixel**2+y_pixel**2)

        max_pixel_distance = evaluate_on_mask_from_list(base_mask, f_max_pixel_distance, [x_pixel_center], theta_range, angles_along_arc_range,
                                   base_pixel_per_segment, func_dimensions=1)

        alpha = np.arccos(max_pixel_distance/r_sun_pixel)
        alpha_dash = np.arccos((max_pixel_distance+ 1/np.sqrt(2))/r_sun_pixel)

        delta_pixel_distance = (alpha-alpha_dash)*r_sun_ref.to_value(u.Mm)

        delta_pixel_distance_staggered_Mm = np.zeros_like(intensity_mean_staggered)
        delta_pixel_distance_staggered_Mm[::ts] = delta_pixel_distance[:,:,:,0]
    elif flat_field:
        delta_pixel_distance_staggered_Mm = None#np.zeros_like(intensity_mean_staggered)*np.nan
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' find_segments_from_list : Warning: As flat_field is enabled,'
                    ' no segment dependent uncerainty is calculated (i.e. delta_pixel_distance_staggered_Mm = None)')
    else:
        delta_pixel_distance_staggered_Mm = None

    for start_index in range(1,ts):
        nr_of_segments = (len(angles_along_arc_range)-1) * (len(theta_range)-1)

        ##############################evaluate Data ######################################################################

        intensity_mean_stag,intensity_median_stag,intensity_var_stag,pixel_per_segment,mask_3_stagg = find_segments_from_list(theta_range,ttheta,angles_along_arc_range+start_index*diff_angles,aangles_along_arc,map_data_list)

        intensity_mean_staggered[start_index::ts,:, :] = intensity_mean_stag[:,:,:]
        intensity_var_staggered[start_index::ts, :, :] = intensity_var_stag[:, :, :]/pixel_per_segment # The variance is given for the peak intensity: var_mean = var/N

        intensity_median_staggered[start_index::ts, :, :] = intensity_median_stag [:, :, :]



        if calculate_uncertainty and (not flat_field):
            max_pixel_distance = evaluate_on_mask_from_list(mask_3_stagg, f_max_pixel_distance, [x_pixel_center],
                                                            theta_range,
                                                            angles_along_arc_range + start_index * diff_angles,
                                                            pixel_per_segment, func_dimensions=1)

            alpha = np.arccos(max_pixel_distance / r_sun_pixel)
            alpha_dash = np.arccos((max_pixel_distance + 1 / np.sqrt(2)) / r_sun_pixel)
            delta_pixel_distance = (alpha - alpha_dash) * r_sun_ref.to_value(u.Mm)

            delta_pixel_distance_staggered_Mm[start_index::ts, :, :] = delta_pixel_distance[:, :, :, 0]

        distance_staggered[start_index::ts] = distance + diff_distance*start_index

        max_pixel_per_segment.append(np.max(pixel_per_segment))
        min_pixel_per_segment.append(np.min(pixel_per_segment))

    #plt.figure()
    #plt.plot(delta_distance_staggered_Mm[:,0,0])
    #plt.show()

    parameter_dict['general'] = ' '
    parameter_dict['general: reference radius sun (in Mm)'] = r_sun_ref.to_value('Mm')
    parameter_dict['general: difference to nominal solar radius (695.7 Mm) (in Mm)'] = r_sun_ref.to_value('Mm')-695.7
    parameter_dict['general: aaa distance between segments unstaggered (in Mm)'] = round(np.diff(distance)[0].to_value(u.Mm),2)
    if flat_field:
        parameter_dict['general: aaa distance between segments unstaggered (in Degree)'] = 'n.A. Flat field calculation'
        parameter_dict['general: aaa max range unstaggered (in Degree)'] = 'n.A. Flat field calculation'
        parameter_dict['general: aaa min range unstaggered (in Degree)'] = 'n.A. Flat field calculation'
    else:
        parameter_dict['general: aaa distance between segments unstaggered (in Degree)'] = round(
        np.diff(angles_along_arc_range)[0] * 180 / np.pi, 2)
        parameter_dict['general: aaa max range unstaggered (in Degree)'] = round(
            angles_along_arc_range[-1] * 180 / np.pi, 2)
        parameter_dict['general: aaa min range unstaggered (in Degree)'] = round(
            angles_along_arc_range[0] * 180 / np.pi, 2)


    parameter_dict['general: aaa max range unstaggered (in Mm)'] = round(distance[-1].to_value(u.Mm),2)
    parameter_dict['general: aaa min range unstaggered (in Mm)'] = round(distance[0].to_value(u.Mm),2)
    parameter_dict['diagnostic: Maximum Nr of pixel per Segment'] = np.max(max_pixel_per_segment)
    parameter_dict['diagnostic: Minimum Nr of pixel per Segment'] = np.min(min_pixel_per_segment)
    parameter_dict['staggered'] = ' '
    parameter_dict['staggered: times staggered'] = times_staggered
    parameter_dict['staggered: distance between staggered segments (in Mm)'] = round(diff_distance.to_value(u.Mm),2)
    parameter_dict['staggered: aaa between staggered segments (in Degree)'] = round(diff_angles* 180 / np.pi,2)

    return intensity_mean_staggered,intensity_median_staggered,intensity_var_staggered,distance_staggered *u.meter,delta_pixel_distance_staggered_Mm,base_mask,base_pixel_per_segment,parameter_dict

@njit(parallel=True)
def extract_loop(segment_values,m_data_list,mask_3_argsort,lower,upper):
        for t in prange(len(m_data_list)):
            segment_values[:,t] = m_data_list[t].flatten()[mask_3_argsort[lower:upper]]


def evaluate_on_mask_from_list(mask_3_full,function,m_data_list,theta_range,angles_along_arc_range,pixel_per_segment,func_dimensions = 1,**kwargs):
    '''Evaluates the function on data of a map sequence using a mask created by the find_segments function.

    :param full_mask:    2d np.array size equal to map size, return of the find_segment function (as maps_data is given by [y,x])
    :param function:                function to evaluate data on f(mat,**kwarg) with mat being [data,time], shall return a np.array with [time, func_dim]
    :param nr_of_segments:          int, number of segments in the mask
    :param m_data:                  3d np.array, data of the map sequence to be evaluated [,,time]
    :param theta_range:             1d np.array, see find_segments function
    :param angles_along_arc_range:  1d np.array, see find_segments function
    :param func_dimensions:         int, nr. of output values of the function
    :param kwargs:                  keyword arguments to be passed to the function
    :return:
        4d np.array with [angles_along_arc-1,theta-1,time,func_dim]
    '''

    time_sequ_len = len(m_data_list)
    nr_of_segments = (len(theta_range) - 1) * (len(angles_along_arc_range) - 1)

    # Maps_data: The first index corresponds to the y direction and the second to the x direction in the two-dimensional pixel coordinate system
    #mask_3_full = full_mask

    #Remove the time axis in pixel per coordinate
    if len(pixel_per_segment.shape) == 2:
        pixel_per_segment = pixel_per_segment[:,0]
    elif len(pixel_per_segment.shape) == 3:
        pixel_per_segment = pixel_per_segment[:, :, 0]

    segment_mat = np.zeros((nr_of_segments, time_sequ_len,func_dimensions))

    #mask_3_sort = np.reshape(mask_3_plot,(mask_3_plot.size))

    mask_3_argsort = np.argsort(mask_3_full, axis = None)
    mask_3_sorted = np.sort(mask_3_full,axis = None)

    #test_dat = np.zeros((m_data_list[0].shape[0]*m_data_list[0].shape[0]))

    for i in range(nr_of_segments):
        #print(f'{i} of {nr_of_segments}')

        segment_values = np.zeros([int(pixel_per_segment.flatten()[i]),time_sequ_len])
        lower = np.searchsorted(mask_3_sorted,i+1,side='left')
        upper = np.searchsorted(mask_3_sorted,i+2,side='left')

        if len(m_data_list) == 1:
            segment_values[:, 0] = m_data_list[0].flatten()[mask_3_argsort[lower:upper]]
        else:
            # The parallelized loop is only used for lists with multiple enteries
            extract_loop(segment_values,m_data_list,mask_3_argsort,lower,upper)
        #for t in range(len(m_data_list)):
        #    segment_values[:,t] = m_data_list[t].flatten()[mask_3_argsort[lower:upper]]

        segment_mat[i, :,:] = function(segment_values,**kwargs) #[segment,time,func_dim]

    result_mat = np.reshape(segment_mat, (len(theta_range) - 1, len(angles_along_arc_range) - 1, len(m_data_list),func_dimensions))
    result_mat = np.einsum('ijkl ->jikl' ,result_mat) #[angles_along_arc,theta,time,func_dim]
    #print('finished function evaluation')

    return result_mat

def find_coord_from_angles(coord_of_flare,
                           theta_mat,
                           arc_angles_mat,
                           flat_field = False,
                           mask_overlimb = True):
    ''' Finds the corresponding x,y,z coordinates of theta and arc angles given in a heliocentric view with respect to
    the coord_of_flare Skycoord object. Takes 2d matrices as input to allow for the calculation of multiple points in
    one function call. For use in plot functions, see the example below.

    :param coord_of_flare:  Skycoord Object of the Flare origin at the intended height
    :param theta_mat:       1d or 2d np.array; theta values with [theta, aaa]
    :param arc_angles_mat:  1d or 2d np.array; aaa   values with [theta, aaa]
    :param flat_field:      bool, switches between flat field at sun distance and following the sun curvature
    :return:
        2d/3d np.array Matrix with values for vectors [xyu,theta,aaa]

    Example:
            coords = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat)

            coords_meter = coords * u.meter
            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky[j, i])
    '''
    import astropy.units as u
    from astropy.coordinates import BaseCoordinateFrame, SkyCoord
    from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, get_body_heliographic_stonyhurst
    import numpy as np

    theta_dims = len(theta_mat.shape)
    distance_unit = u.m  # Define meter as the unite that is used #start.cartesian.xyz.unit

    start = coord_of_flare.transform_to(Heliocentric)

    center = SkyCoord(0 * distance_unit,
                      0 * distance_unit,
                      0 * distance_unit,
                      obstime=start.obstime,
                      observer=start.observer,
                      frame=Heliocentric)

    start_cartesian = start.cartesian.xyz.to(distance_unit).value

    if not(flat_field):

        #end_cartesian = end.cartesian.xyz.to(start.distance_unit).value
        center_cartesian = center.cartesian.xyz.to(distance_unit).value

        v1 = start_cartesian - center_cartesian
        r_sun = np.linalg.norm(v1)

        # Defines the second Vector to point to the north pole
        # Form Heliocentric coordinates:
        # The Y-axis is aligned with the component of the vector to the Sun’s north pole that is perpendicular to the Z-axis.
        v2 = np.array([0,r_sun,0])

        # Initial v3 vektor pointing north and perpendicular to v1
        v3 = np.cross(np.cross(v1, v2), v1)
        v3 = r_sun * v3 / np.linalg.norm(v3)

        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        # Temporary Vector needed for the rotating vector
        v4 = np.cross(v1,v3)/r_sun

        # Using einsum to creat the matrices
        # Note on the notation: the number of the first letter indicates the dimensions

        if theta_dims == 2:
            vvv1 = np.einsum('i,jk -> ijk', v1, np.ones_like(theta_mat))
            vvv3 = np.einsum('i,jk -> ijk', v3, np.ones_like(theta_mat))
            vvv4 = np.einsum('i,jk -> ijk', v4, np.ones_like(theta_mat))
            tttheta = np.einsum('i,jk -> ijk',  np.ones_like(v3),theta_mat)
            aaarcangles = np.einsum('i,jk -> ijk',  np.ones_like(v3), arc_angles_mat)
        if theta_dims == 1:
            vvv1 = np.einsum('i,j -> ij', v1, np.ones_like(theta_mat))
            vvv3 = np.einsum('i,j -> ij', v3, np.ones_like(theta_mat))
            vvv4 = np.einsum('i,j -> ij', v4, np.ones_like(theta_mat))
            tttheta = np.einsum('i,j -> ij',  np.ones_like(v3),theta_mat)
            aaarcangles = np.einsum('i,j -> ij',  np.ones_like(v3), arc_angles_mat)

        vvv_rot = vvv3*np.cos(tttheta) + vvv4 * np.sin(tttheta) #+ v1*np.dot(v1,v3) / r_sun**2 * (1-np.cos(theta)) #The last part is redundant as <v1|v3> = 0

        gggreat_arc_points_carthesian = vvv1 *np.cos(aaarcangles) + vvv_rot * np.sin(aaarcangles)

        # Added an "mask overlimb" keyword to the find_coord_from_angles function. If false,
        # vectors with negative z values (i.e. over the limb) are not replaced with nan.
        if mask_overlimb:
            if theta_dims == 2:
                mask = gggreat_arc_points_carthesian[2,:,:] < 0
            if theta_dims == 1:
                mask = gggreat_arc_points_carthesian[2, :] < 0
            gggreat_arc_points_carthesian[0,mask] = np.nan
            gggreat_arc_points_carthesian[1, mask] = np.nan
            gggreat_arc_points_carthesian[2, mask] = np.nan

        gggreat_arc_points_carthesian = gggreat_arc_points_carthesian * distance_unit
    else:
        flare_pos_xyz_Mm = coord_of_flare.transform_to(Heliocentric).cartesian.xyz.to_value('Mm')

        # Adds to the flare position in cartesian coordinates the arc angles (arc angles in Mm)
        x_distance = np.cos(theta_mat+np.pi/2)*arc_angles_mat +flare_pos_xyz_Mm[0]
        y_distance = np.sin(theta_mat + np.pi / 2) * arc_angles_mat +flare_pos_xyz_Mm[1]

        z_distance = np.zeros_like(x_distance)



        gggreat_arc_points_carthesian = np.stack((x_distance,y_distance,z_distance),axis = 0) *u.Mm


    return gggreat_arc_points_carthesian



############################################################################################
#
# Distance Uncertainty Calculation
#
#############################################################################################

def distance_uncertainties(distance,
                           times_staggered,
                           r_sun_ref,
                           r_sun_pixel,
                           intensity_median,
                           intensity_var,
                           delta_distance_pixel = None,
                           flat_field = False):
    """
    Calculated the upper and lower uncertainty in distance including "interpolated amplitude uncertainty" and
    "pixel error" for the front and tracing edge.
    Calculates the

    :param distance:        1d np.array * distance_unit, distance of the staggered segments
    :param times_staggered: 1d list, times of the observation
    :param r_sun_ref:       float * distance_unit, reference radius for the calculations
    :param intensity_median: see find_segments_from_list_staggered
    :param intensity_var:   see find_segments_from_list_staggered
    :param delta_distance_pixel: 1d np.array or None,
                                   1d is output from find_segments_from_list_staggered(calculate_uncertainty =True);
                                   None calculates pixel uncertainty as constant value at 70°
    :return:
        delta_distance:     4d np.array, upper/lower interpolated amplitude uncertainty + pixel error in (Mm)
                                         [upper/lower,angles_along_arc, theta, time]
        segment_pixel_uncertainty: 3d np.array, pixel + segment error for segment [angles_along_arc, theta, time]
        delta_d_segment_size:      float, segment size error to be added to delta_distance in the peak finder
        segment_length:            float, length of a segment in Mm
    """

    ##########################################
    # Segment Uncertainty
    ##########################################

    distance_MM = distance.to_value('Mm')
    diff_distance_MM = (distance_MM[1] - distance_MM[0])

    # Segment length
    segment_length = diff_distance_MM * times_staggered

    # Segment distance Error
    delta_d_segment_size = segment_length/2+diff_distance_MM



    ##########################
    #Pixel error
    ##########################
    if delta_distance_pixel is None:
        r_sun = r_sun_ref.to_value(u.m)
        l_pixel = r_sun / r_sun_pixel

        if flat_field:
            delta_d_pixel_size = np.sqrt(2)/2*l_pixel*u.m
        else:
        # Calculated for 70° from Center
            x_min = r_sun*np.sin(np.pi*70/180)
            x_delta_x_max = r_sun*np.sin(np.pi*70/180) + np.sqrt(2)/2 *l_pixel

            alpha_pix_err = np.arccos(x_min/r_sun)
            alpha_dash_pix_err = np.arccos(x_delta_x_max / r_sun)

            delta_d_pixel_size= r_sun * (alpha_pix_err - alpha_dash_pix_err)#/2 #

            delta_d_pixel_size = delta_d_pixel_size*u.m
            print('pixel size Uncertainty %.2f' %(delta_d_pixel_size.to_value('Mm')))

    ########################
    # Interpolated amplitude uncertainty to distance uncertainty
    #######################
    intensity_std = np.sqrt(intensity_var)

    distance_MM_Mat = np.einsum('i,ijk->ijk',distance_MM,np.ones_like(intensity_median))

    intensity_upper = intensity_median + intensity_std
    intensity_lower = intensity_median - intensity_std

    k_upper = np.diff(intensity_upper,axis=0)/np.diff(distance_MM_Mat,axis=0)
    k_lower = np.diff(intensity_lower, axis=0) / np.diff(distance_MM_Mat, axis=0)

    delta_upper_right = -intensity_std[:-1,:,:]/k_upper
    delta_lower_right = intensity_std[:-1, :, :] / k_lower

    delta_upper_left = -intensity_std[1:,:,:]/k_upper
    delta_lower_left = intensity_std[1:, :, :] / k_lower


    delta_right = np.where(delta_upper_right > 0,delta_upper_right,delta_lower_right)
    delta_right[delta_right<0] = diff_distance_MM
    delta_right[delta_right>diff_distance_MM] = diff_distance_MM

    delta_left  = np.where(delta_lower_left  < 0,delta_lower_left,delta_upper_left)
    delta_left[delta_left>0] = -diff_distance_MM
    delta_left[delta_left<-diff_distance_MM] = -diff_distance_MM


    # For the right most value no distance Uncertainty based on Amplitude is assigned in the positive direction

    if delta_distance_pixel is None:
        delta_distance = np.zeros((2, intensity_median.shape[0], intensity_median.shape[1], intensity_median.shape[2]))

        # Note: The segment uncertainty is added in the Peak_finding_algorithm, as this allows the application
        #        of asymmetric uncertainties depending on front and trailing edge
        #        i.e. the delta_d_segment_uncertainty is only added to the front facing uncertainty of a front point
        delta_distance[1,:-1,:,:] = delta_right  + delta_d_pixel_size.to_value('Mm') #+ delta_d_segment_size
        delta_distance[0,1:,:,:] = -delta_left  + delta_d_pixel_size.to_value('Mm') #+ delta_d_segment_size

        segment_pixel_uncertainty = delta_d_segment_size + delta_d_pixel_size.to_value('Mm')*np.ones_like((intensity_median))
    else:
        delta_distance = np.zeros((2, intensity_median.shape[0], intensity_median.shape[1], intensity_median.shape[2]))

        # Note: The segment uncertainty is added in the Peak_finding_algorithm, as this allows the application
        #        of asymmetric uncertainties depending on front and trailing edge
        #        i.e. the delta_d_segment_uncertainty is only added to the front facing uncertainty of a front point
        delta_distance[1,:,:,:] = delta_distance_pixel[:,:,:] #+ delta_d_segment_size
        delta_distance[0,:,:,:] = delta_distance_pixel[:,:,:]  # + delta_d_segment_size

        delta_distance[1, :-1, :, :] =delta_right  + delta_distance[1, :-1, :, :]
        delta_distance[0, 1:, :, :] = - delta_left + delta_distance[0, 1:, :, :]

        segment_pixel_uncertainty = delta_d_segment_size + delta_distance_pixel[:,:,:]

    # Set delta distance of first point to nan if point is nan
    delta_distance[0, 0, :, :][np.isnan(intensity_median[0, :, :])] = np.nan
    # Set delta distance of last point to nan if point is nan
    delta_distance[1, -1, :, :][np.isnan(intensity_median[-1, :, :])] = np.nan

    return delta_distance,segment_pixel_uncertainty,delta_d_segment_size,segment_length



############################################################################################
#
# Peak fitting and wave tracing
#
#############################################################################################


#@njit(parallel=True)
def peak_finding_algorithm(intensity_mean,
                           intensity_std,
                           theta_range,
                           distance,
                           delta_distance,
                           segment_pixel_uncertainty,
                           delta_d_segment_size,
                           max_nr_peaks_const=5,
                           wavefront_cutof=0.5,
                           cutoff_type = 'relative',
                           min_peak_height=1.1,
                           c_closest=0.03,
                           sectorwarnings_active = True,
                           parameter_dict={}):
    ''' Finds the peaks in the perturbation profiles given by intensty_mean and distance. Uses custom parameters to
    allow fine adjustment

    :param intensity_mean:      3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
    :param intensity_std:       3d np.array, standard deviation (= sqrt of variance) of the intensity of each segment [angles_along_arc, theta, time]
    :param theta_range:         1d np.array, vector of the segment borders along ttheta direction
    :param distance:            1d np.array, Distance in length units (e.g. [1e6,2e6] * u.m) of astropy.units
    :param delta_distance:      4d np.array, upper/lower interpolated amplitude uncertainty + pixel error in (Mm)
                                             [upper/lower,angles_along_arc, theta, time]
    :param segment_pixel_uncertainty: 3d np.array, pixel + segment error for segment [angles_along_arc, theta, time]
    :param delta_d_segment_size:      float, segment size error to be added to delta_distance in the peak finder
    :param time:                list with strings, time string of the observation of images
    :param max_nr_peaks_const:  int                 ; upper Limit of peaks searched for
    :param wavefront_cutof:     np.float: [% of peak];  Percent of peak height defining the wavefront/trail
    :param min_peak_height:     np.float: [%]       ; Minimum peak height required
    :param c_closest:           np.float: [%]       ; Prominence, height between new peak to minimum to the closest other peak
    :param parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        d_peak_mat      3d np.array, distance of peaks to wave origin [theta,time,peak_nr]
        d_front_mat     3d np.array, distance of front edges to wave origin [theta,time,peak_nr]
        d_trail_mat     3d np.array, distance of trailing edges to wave origin [theta,time,peak_nr]
        peak_mat        3d np.array, amplitude of peaks  [theta,time,peak_nr]
        front_mat       3d np.array, amplitude of front edges  [theta,time,peak_nr]
        trail_mat       3d np.array, amplitude of trailing edges  [theta,time,peak_nr]
        delta_peak_mat  3d np.array, uncertainty of the peaks amplitude to wave origin [theta,time,peak_nr]
        t_sunpy_sec     1d np.array, sunpy time object in seconds
        max_nr_peaks_vec        1d np.array, maximum nr of peaks-found-per-timestep, in each sector [theta]
        max_nr_peaks_const      int, maximum nr of peaks searched for
                parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions

    '''

    from scipy.optimize import curve_fit

    def gaussian(x, A, x0, sigma1):#,sigma2):
        #return np.where(x <= x0,
        #                1 + A * np.exp(-(x - x0) ** 2 / (2 * sigma1 ** 2)),
        #                1 + A * np.exp(-(x - x0) ** 2 / (2 * sigma2 ** 2)))
        return 1 + A * np.exp(-(x - x0) ** 2 / (2 * sigma1 ** 2))

    assert len(intensity_mean.shape) == 3, "intensity_mean requires 3 dimensions [angles_along_arc, theta, time]"
    assert len(intensity_std.shape) == 3, "intensity_sqrt requires 3 dimensions [angles_along_arc, theta, time]"
    #assert intensity_mean.shape == intensity_std.shape, "intensity_mean and intensity_staggered must have the same shape"
    assert len(distance) == intensity_mean.shape[
        0], "distance needs to have the same length as axis 0 of intensity_mean"
    assert theta_range.shape[0] - 1 == intensity_mean.shape[
        1], "theta_range must be shorter by 1 value than the axis 1 of intensity_mean"


    # Matrices Saving the distance along the solar surface from its origin
    d_peak_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    d_fitted_peak_mat = np.zeros_like(d_peak_mat) * np.nan
    d_front_mat = np.zeros_like(d_peak_mat) * np.nan
    d_trail_mat = np.zeros_like(d_peak_mat) * np.nan


    # Matrices Saving the hight value of the wave
    peak_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    fitted_peak_mat = np.zeros_like(peak_mat) * np.nan
    front_mat = np.zeros_like(peak_mat) * np.nan
    trail_mat = np.zeros_like(peak_mat) * np.nan

    # Matrix saving the Sigma of the gaussian Peak fit.
    sig_fitted_peak_mat = np.zeros_like(peak_mat) * np.nan


    # Uncertainty Matrices
    delta_peak_mat = np.zeros_like(peak_mat)*np.nan
    delta_fitted_peak_mat = np.zeros_like(peak_mat)*np.nan
    #delta_front_mat = np.zeros_like(peak_mat) * np.nan
    #delta_trail_mat = np.zeros_like(peak_mat) * np.nan

    # Uncertainty Distance Mat
    delta_d_peak_mat = np.zeros((2,d_peak_mat.shape[0],d_peak_mat.shape[1],d_peak_mat.shape[2]))*np.nan
    delta_d_fitted_peak_mat = np.zeros_like(delta_d_peak_mat)*np.nan
    delta_d_trail_mat = np.zeros_like(delta_d_peak_mat)*np.nan
    delta_d_front_mat = np.zeros_like(delta_d_trail_mat)*np.nan

    #############
    # Add Segment Uncertainty
    ##########
    # NOTE: Despite the option to add delta_d_segment_size based on the direction of the uncertainty with respect to
    #       the wave (i.e. front-phasing uncertainty of front is treated differently as trail-phasing uncertainty of front)
    #       this is not implemented and the uncertainty is applied symmetrically
    delta_distance = delta_distance + delta_d_segment_size
    segment_pixel_uncertainty_with_nan = np.copy(segment_pixel_uncertainty)

    max_nr_peaks_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2]),
                                dtype=np.int64)  # Maximum number of Peaks found in each theta and timestep

    if cutoff_type.lower() == 'relative':
        if wavefront_cutof > 0.5:
            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' peak_finding_algorithm: Warning: The fit range of the gaussian (FWHM) is larger than the wavefront_cutof')
    elif cutoff_type.lower() == 'absolut':
        if wavefront_cutof < ((min_peak_height-1)*0.5)+1:
            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' peak_finding_algorithm: Warning: The fit range of the gaussian (FWHM) is larger than the wavefront_cutof')
        assert min_peak_height > wavefront_cutof,'the minimum peak height is smaller than the wavefront_cutof'
    else:
        assert False, ' peak_finding_algorithm: Unknown type for cutoff_type, chose "relative" or "absolut" '

    ##################################################################################
    # Peak Finding Algorithm
    ##################################################################################

    # Integrating over different sectors (e.g. theta range)
    for j in range(intensity_mean.shape[1]):
        # Check if there are nans in the vector indicating a sector over the horizion

        #assert not np.any(np.isnan(intensity_mean[:, j, :])), ("The sector with index j = %.0f contains nan values, "
        #                                                       "likely due to angle_along_arc reaching over the solar "
        #                                                       "horizon" %(j))
        if np.any(np.isnan(intensity_mean[:, j, :])) and sectorwarnings_active:
            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' peak_finding_algorithm: Warning: The Sector includes nan values in intensity_mean')

        distance_MM_with_nan = distance.to_value(u.Mm)
        diff_distance_MM = (distance_MM_with_nan[1] - distance_MM_with_nan[0])

        theta_angle = (theta_range[j] + theta_range[j + 1]) / 2

        peaks_in_segment = False

        for t in range(intensity_mean.shape[2]):
            mean_int_vector = np.copy(intensity_mean[:, j, t])
            std_int_vector = np.copy(intensity_std[:, j, t])

            not_nan_mask_ = ~np.isnan(mean_int_vector)
            not_zero_mask_ = ~(std_int_vector == 0)
            not_nan_zero_mask = not_nan_mask_ & not_zero_mask_

            mean_int_vector = mean_int_vector[not_nan_zero_mask]
            mean_int_vector[mean_int_vector < 1] = 1

            std_int_vector = std_int_vector[not_nan_zero_mask]

            distance_MM = distance_MM_with_nan[not_nan_zero_mask]
            segment_pixel_uncertainty = segment_pixel_uncertainty_with_nan[not_nan_zero_mask]

            delta_d_vector = np.copy(delta_distance[:,:,j,t])
            delta_d_vector = delta_d_vector[:,not_nan_zero_mask]



            # find all lokal maxima
            lok_max_arg = find_peaks(mean_int_vector)[0]
            lok_min_arg = find_peaks(-mean_int_vector)[0]

            #Peaks that are below a certain threshold are excempted
            lok_max_arg = lok_max_arg[mean_int_vector[lok_max_arg] >= min_peak_height]

            if lok_max_arg.shape[0] != 0:
                lok_max = mean_int_vector[lok_max_arg]

                peak = np.max(lok_max)
                peak_arg = lok_max_arg[lok_max == peak][0]

                front_args_vec = np.zeros((max_nr_peaks_const + 1))  # Allways has the border argument as a value
                boxing_front_arg = 0
                # d_front_vec = np.ones_like(front_args_vec)
                trail_args_vec = np.ones((max_nr_peaks_const + 1)) * len(
                    mean_int_vector)  # Allways has the border argument as a value
                boxing_trail_arg = len(mean_int_vector)

                peaks_in_segment = True

                for i in range(max_nr_peaks_const):
                    if (lok_max.shape[0] != 0):
                        max_nr_peaks_mat[j, t] = i + 1

                        d_peak_mat[j, t, i] = distance_MM[peak_arg]
                        peak_mat[j, t, i] = peak
                        delta_peak_mat[j, t, i] = std_int_vector[peak_arg]

                        # Interpolates the segment dependent uncertainty of pixel error and segment error and evaluates
                        # it on the peak distance d_peak_mat[j, t, i]
                        delta_d_peak_mat[:,j,t,i] = np.interp(d_peak_mat[j, t, i],distance_MM,segment_pixel_uncertainty[:,j,t])

                        if cutoff_type.lower() == 'relative':
                            cutoff_height = (peak - 1) * wavefront_cutof + 1
                        elif cutoff_type.lower() == 'absolut':
                            cutoff_height = wavefront_cutof

                        #cutoff_height = (min_peak_height - 1) * wavefront_cutof + 1 # The wavefront is allways at the same height given by the min peak height
                        cutoff_FWHM = (peak - 1) * 0.5 + 1

                        smaller_cutoff_arg = np.where(cutoff_height > mean_int_vector)[0]
                        arg_diff = (peak_arg - smaller_cutoff_arg)

                        try:
                            trail_arg = np.max(smaller_cutoff_arg[arg_diff > 0])

                            if trail_arg < boxing_front_arg:  # If there is another front within the trail of the peak, seek min between other front and current peak
                                trail_arg = int(boxing_front_arg + np.argmin(mean_int_vector[int(boxing_front_arg):peak_arg]))

                                # Include the corner piller in the vector (e.g. 2,5,8,6,2) = > 5 is the trail corner
                                trail_args_vec[i] = trail_arg + 1
                                trail_arg_for_fit = trail_arg + 1

                                d_trail_mat[j, t, i] = distance_MM[trail_arg]
                                delta_d_trail_mat[:,j, t, i] = delta_d_vector[:,trail_arg]
                                trail_mat[j, t, i] = mean_int_vector[trail_arg]
                            else:
                                trail_args_vec[
                                    i] = trail_arg + 1  # Include the corner piller in the vector (e.g. 2,5,8,6,2) = > 5 is the trail corner
                                trail_arg_for_fit = trail_arg + 1

                                if np.abs((mean_int_vector[trail_arg + 1] - mean_int_vector[trail_arg])) < 1e-10:
                                    factor_trail = 0.5 #If both values are the same, the wave trail is assumed to be in the middle
                                else:
                                    factor_trail = (1/ (mean_int_vector[trail_arg + 1] - mean_int_vector[trail_arg])
                                                    * (cutoff_height - mean_int_vector[trail_arg]))

                                # Interpolating
                                d_trail_mat[j, t, i] = (distance_MM[trail_arg]
                                                        + (distance_MM[trail_arg + 1] - distance_MM[trail_arg])
                                                        * factor_trail)
                                delta_d_trail_mat[:,j, t, i] = (delta_d_vector[:,trail_arg+1] * factor_trail
                                                                + delta_d_vector[:,trail_arg]*(1-factor_trail))

                                trail_mat[j, t, i] = cutoff_height
                        except:

                            # Checks if there is another front between the current peak and the trail-border
                            if boxing_front_arg != 0:  # If there is another front within the trail of the peak, seek min between other front and current peak
                                trail_arg = int(boxing_front_arg + np.argmin(mean_int_vector[int(boxing_front_arg):peak_arg]))

                                # Include the corner piller in the vector (e.g. 2,5,8,6,2) = > 5 is the trail corner
                                trail_args_vec[i] = trail_arg + 1
                                trail_arg_for_fit = trail_arg + 1

                                d_trail_mat[j, t, i] = distance_MM[trail_arg]
                                delta_d_trail_mat[:, j, t, i] = delta_d_vector[:, trail_arg]
                                trail_mat[j, t, i] = mean_int_vector[trail_arg]

                            else:
                                trail_arg = 0
                                trail_arg_for_fit = 0

                            #print('no trail found')
                        try:
                            front_arg = np.min(smaller_cutoff_arg[arg_diff < 0])

                            if front_arg > boxing_trail_arg:  # If there is another trail within the front of the peak, seek min between other trail and current peak
                                front_arg = int(peak_arg + np.argmin(mean_int_vector[peak_arg:int(boxing_trail_arg)]))

                                # Include the corner piller in the vector,(e.g. 2,5,8,6,2) = > 6 is the front corner
                                front_args_vec[i] = front_arg - 1

                                d_front_mat[j, t, i] = distance_MM[front_arg]
                                delta_d_front_mat[:, j, t, i] = delta_d_vector[:, front_arg]
                                front_mat[j, t, i] = mean_int_vector[front_arg]
                            else:
                                # Include the corner piller in the vector,(e.g. 2,5,8,6,2) = > 6 is the front corner
                                front_args_vec[i] = front_arg - 1

                                # Interpolating
                                if np.abs((mean_int_vector[front_arg] - mean_int_vector[front_arg - 1])) < 1e-10:
                                    factor = 0.5 #If both values are the same, the wave front is assumed to be in the middle
                                else:
                                    factor = (1 / (mean_int_vector[front_arg] - mean_int_vector[front_arg - 1])
                                              * (cutoff_height - mean_int_vector[front_arg - 1]))

                                d_front_mat[j, t, i] = distance_MM[front_arg - 1] + (
                                            distance_MM[front_arg] - distance_MM[front_arg - 1]) * factor
                                delta_d_front_mat[:,j, t, i] = (delta_d_vector[:, front_arg]*factor
                                                                + delta_d_vector[:, front_arg-1]*(1-factor))
                                front_mat[j, t, i] = cutoff_height
                        except:
                            # Checks if there is another peak between the front-border and the current peak
                            if boxing_trail_arg != len(mean_int_vector):
                                front_arg = int(peak_arg + np.argmin(mean_int_vector[peak_arg:int(boxing_trail_arg)]))

                                # Include the corner piller in the vector,(e.g. 2,5,8,6,2) = > 6 is the front corner
                                front_args_vec[i] = front_arg - 1

                                d_front_mat[j, t, i] = distance_MM[front_arg]
                                delta_d_front_mat[:, j, t, i] = delta_d_vector[:, front_arg]
                                front_mat[j, t, i] = mean_int_vector[front_arg]
                            else:
                                front_arg = len(mean_int_vector)
                                if sectorwarnings_active:
                                    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                                    print(now + ' peak_finding_algorithm: no front found at %.0f° theta ' % (
                                        theta_angle) + f'and time_index: {t}')

                        #####################################
                        # Peak fitting
                        #####################################
                        # As for vec = [1,2,3,4], vec[0:2] = [1,2] not includes 3, the front_args_vec +1 = front_args is
                        # used to exclude the pillar, but include all others
                        fit_vector = mean_int_vector[trail_arg_for_fit:front_arg]
                        fit_distance = distance_MM[trail_arg_for_fit:front_arg]
                        delta_fit_vector = std_int_vector[trail_arg_for_fit:front_arg]

                        try:


                            mu_guess = np.sum(fit_vector*fit_distance)/np.sum(fit_vector)                   #sum(pi/sum(pi) * xi)= sum (pi x_i)/sum(pi)
                            ##var_guess = np.sum(fit_vector*fit_distance**2)/(np.sum(fit_vector))-mu_guess**2 #<xi**2> - mu**2 = sum( pi/sum(pi) * xi**2)-mu**2
                            sigma_guess = distance_MM[front_arg-1]-distance_MM[trail_arg+1]
                            sigma_trail_guess = sigma_guess#2*(mu_guess-distance_MM[trail_arg])
                            sigma_front_guess = sigma_guess#2*(distance_MM[front_arg]-mu_guess)
                            peak_guess = peak+0.1

                            # The number of datapoints given needs to be larger than the number of parameters
                            # (Equal might also suffice)
                            if len(fit_vector) > 3:
                                ## gaussian(x, H, A, x0, sigma):
                                ## y =  H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

                                #Requirements for Bounds:
                                # - peak: should not be significantly bigger than the maximum,
                                #         so 2 the maximum might be already excessive
                                # - mu: must be between front and trail
                                # - sigma: the least constrained parameter, 10 times the peak width should give sufficient
                                #          room in the case of a very flat distribution

                                a,cov= curve_fit(gaussian,
                                                 fit_distance,
                                                 fit_vector,
                                                 p0 = [peak_guess-1,mu_guess,sigma_guess],
                                                 bounds=[[0,distance_MM[trail_arg+1],0],
                                                         [peak_guess*2,distance_MM[front_arg-1],10*sigma_guess]],
                                                 sigma = delta_fit_vector,
                                                 )
                                                 #jac='cs') #Did not improve fitting
                                                 #absolute_sigma=True) #

                                d_fitted_peak_mat[j, t, i] = a[1]
                                fitted_peak_mat[j, t, i] = a[0] + 1
                                sig_fitted_peak_mat[j, t, i] = a[2]

                                delta_d_fitted_peak_mat[:,j, t, i] = np.sqrt(cov[1,1]) + np.interp(a[1],distance_MM,segment_pixel_uncertainty[:,j,t])

                                # Check if Uncertainty range of Peak is within Peak range
                                peak_range_front = (d_front_mat[j,t,i]-d_fitted_peak_mat[j, t, i]+delta_d_front_mat[1, j, t, i])
                                peak_range_trail = (d_fitted_peak_mat[j, t, i]-d_trail_mat[j,t,i]+delta_d_trail_mat[0, j, t, i])

                                # Cut uncertainty of fitted peak if outside of peak range
                                if delta_d_fitted_peak_mat[1,j, t, i] > peak_range_front:
                                    delta_d_fitted_peak_mat[1, j, t, i] = peak_range_front
                                if delta_d_fitted_peak_mat[0,j, t, i] > peak_range_trail:
                                    delta_d_fitted_peak_mat[0, j, t, i] = peak_range_trail

                                delta_fitted_peak_mat[j,t,i] = np.sqrt(cov[0,0])

                            else:
                                d_fitted_peak_mat[j, t, i] = distance_MM[peak_arg]
                                fitted_peak_mat[j, t, i] = peak

                                peak_range_front = (d_front_mat[j,t,i]-d_fitted_peak_mat[j, t, i]+delta_d_front_mat[1, j, t, i])
                                peak_range_trail = (d_fitted_peak_mat[j, t, i]-d_trail_mat[j,t,i]+delta_d_trail_mat[0, j, t, i])

                                # In the case of no fit, the uncertainty is set to its local maximum
                                delta_d_fitted_peak_mat[1,j, t, i] = peak_range_front
                                delta_d_fitted_peak_mat[0, j, t, i] = peak_range_trail

                                delta_fitted_peak_mat[j, t, i] = std_int_vector[peak_arg]

                        except:
                            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                            print(now + ' peak_finding_algorithm: fitting a peak failed at %.0f° theta ' % (theta_angle) + f'and time_index: {t}')

                            # Possible sources of failure:
                            # - to little data points: for 3 parameters at least 3 independent variables are needed



                        #################################
                        # Preparing for next peak search
                        #################################

                        mask = np.ones_like(lok_max, dtype=bool)
                        mask[(lok_max_arg > trail_arg) & (lok_max_arg < front_arg)] = False
                        # np.where((not((lok_max_arg>trail_lok_min_arg)&lok_max_arg<front_lok_min_arg)),lok_max,None)

                        lok_max = lok_max[mask]
                        lok_max_arg = lok_max_arg[mask]

                        peak_valid = False

                        while ((lok_max.shape[0] != 0) & (peak_valid == False)):
                            peak = np.max(lok_max)
                            peak_arg = lok_max_arg[lok_max == peak][0]

                            boxing_front_arg = np.max(front_args_vec[front_args_vec < peak_arg])
                            boxing_trail_arg = np.min(trail_args_vec[trail_args_vec > peak_arg])

                            #Checks if it makes sense to look for a trail prominence
                            if int(boxing_front_arg) != 0:
                                h_diff_front = peak - np.min(mean_int_vector[int(boxing_front_arg):peak_arg])
                                d_diff_front = distance_MM[peak_arg] - distance_MM[
                                    int(boxing_front_arg + 1)]  # The one is to correct for the inclusion of the pillar
                            else:
                                # The boxing_front_arg is 0, the h_diff_front value is set to large enough
                                h_diff_front = c_closest+1

                            # Checks if it makes sense to look for a front prominence
                            if int(boxing_trail_arg) != len(mean_int_vector):
                                h_diff_trail = peak - np.min(mean_int_vector[peak_arg:int(boxing_trail_arg)])
                                d_diff_trail = distance_MM[int(boxing_trail_arg - 1)] - distance_MM[
                                    peak_arg]  # The one is to correct for the inclusion of the pillar
                            else:
                                # The boxing_trail_arg is the length of the vector,
                                # the h_diff_front value is set to large enough
                                h_diff_trail = c_closest+1

                            ############################################################
                            # Implementation of c as a % height * distance parameter
                            ##########################################################
                            #if (h_diff_trail * d_diff_trail > c_closest) & (
                            #        h_diff_front * d_diff_front > c_closest):
                                # Peak is valid if its product of distance and prominance is above a certain threshold
                            #    peak_valid = True

                            # Implementation of c as % height prominence within the closest neighbours
                            if (h_diff_trail > c_closest) & (h_diff_front > c_closest):
                                # Peak is valid if its product of distance and prominance is above a certain threshold
                                peak_valid = True

                            else:
                                # Otherwise, peak gets removed from list
                                lok_max = lok_max[lok_max_arg != peak_arg]
                                lok_max_arg = lok_max_arg[lok_max_arg != peak_arg]


                if lok_max.shape[0] != 0:
                    print('More than %.0f peaks found in %.0f° theta at ' % (max_nr_peaks_const, theta_angle) + f'and time_index: {t}')

    #############################################
    # Addition of the Segment Error
    # NOTE: Outdated, is now added to the input values
    ############################################
    # Asymmetric Error for trailing/leading edge only valid if the absolute front/trailheigth is very close to "1"
    #delta_d_front_mat[0,:,:,:] = delta_d_front_mat[0,:,:,:] + delta_segment
    #delta_d_trail_mat[1,:,:,:] = delta_d_trail_mat[1,:,:,:] + delta_segment

    # For heights closer to the FWHM, the uncertainty has to be assumed symmetric
    #delta_d_front_mat[:,:,:,:] = delta_d_front_mat[:,:,:,:] + delta_segment
    #delta_d_trail_mat[:,:,:,:] = delta_d_trail_mat[:,:,:,:] + delta_segment

    max_nr_peaks_vec = np.nanmax(max_nr_peaks_mat, axis=1)

    #parameter_dict['peak finding algorithm: Limit of peaks searched for'] = max_nr_peaks_const
    #parameter_dict['peak finding algorithm: Percent of peak height defining the wavefront/trail'] = wavefront_cutof
    #parameter_dict['peak finding algorithm: Minimum peak height required'] = min_peak_height
    #parameter_dict['peak finding algorithm: height between new peak to minimum to colosest other peak * distance new peak to other front/trail (% * mM)'] = c_closest

    parameter_dict['peak finding algorithm'] = ' '
    parameter_dict['peak finding algorithm: max_nr_peaks_const'] = max_nr_peaks_const
    parameter_dict['peak finding algorithm: wavefront_cutof (% of peak height)'] = wavefront_cutof
    parameter_dict['peak finding algorithm: min_peak_height (%)'] = min_peak_height
    #parameter_dict['peak finding algorithm: c_closest (% * mM)'] = c_closest # Not used
    parameter_dict['peak finding algorithm: c_closest (%)'] = c_closest

    return (d_peak_mat,d_front_mat, d_trail_mat,d_fitted_peak_mat,
            peak_mat,fitted_peak_mat, front_mat, trail_mat,
            sig_fitted_peak_mat,
            delta_peak_mat,delta_fitted_peak_mat,
            delta_d_peak_mat,delta_d_fitted_peak_mat,delta_d_front_mat,delta_d_trail_mat,
            max_nr_peaks_vec, max_nr_peaks_const, parameter_dict)


def wave_tracing_algorithm(d_fitted_peak_mat,
                           delta_d_fitted_peak_mat,
                           d_front_mat,
                           delta_d_front_mat,
                           t_sunpy_sec,
                           max_nr_peaks_vec,
                           peak_tracked=True,
                           v_min_step=10,
                           v_max_step=2000,
                           min_points_in_wave='default',
                           fit_second_feature=False,
                           parameter_dict={}):
    ''' Traces the waves in the output of the peak_finding_algorithm. Can in principle work with any feature (peak,
    front edge, or custom combination of both). A linear fit is applied to all waves detected. The function allows
    the fitting of a secondary feature based on waves detected for the primary one.

    :param d_wave_mat:          3d np.array, distance of feature tracked, [theta,time,peak_index]
    :param d_wave_std_mat:      3d np.array, std of the d_wave_mat parameters
    :param t_sunpy_sec:         1d np.array, sunpy time object in seconds
    :param max_nr_peaks_vec:    1d np.array, maximum nr of peaks-found-per-timestep, in each sector [theta]
    :param v_min_step:          float, minimum velocity of waves between time steps accepted (km/s)
    :param v_max_step:          float, maximum velocity of waves between time steps accepted (km/s)
    :param min_points_in_wave:  string or int, minimum number of time steps with wave detections to accept a new wave
                                default: 'default' => nr is 4 for with average delta_t between observations of less than 60 s
                                                   => nr is 3 for all with more than 60 s
    :param fit_second_feature:  Bool, enables fitting of a secondary feature (e.g. trail if peak is the main feature)
    :param d_feature2_mat:      3d np.array, distance of secondary feature, [theta,time,peak_index]
    :param d_feature2_std_mat:  3d np.array, std of the of secondary feature
    :param parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        wave_value_dict:    dictionary of the results of the wave tracing algorithm
        most important listed:
            waves_feature_index     1d np.arrays in list of lists, indices of the d_wave_mat corresponding to a wave [theta][wave_nr]
            waves_time_index        1d np.arrays in list of lists, indices of the time_mat corresponding to a wave [theta][wave_nr]
            waves_fit               1d np.arrays in list of lists, fit parameters corresponding to a wave [theta][wave_nr]
            waves_fit_cov           2d np.arrays inlist of lists, covariants of fit parameters corresponding to a wave [theta][wave_nr]
            wave_ind_mat_2          3d np.array, matrix with entries of wave_nr on the position corresponding  with features
                                                 in the d_wave_mat, 0 at every position without a wave association [theta,time,peak_index]
        parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    '''
    ########################################################
    # Addition 27.02.2026: Handles Front and Peak Tracking internally
    ################################################################################
    if peak_tracked:
        d_wave_mat = d_fitted_peak_mat
        d_wave_std_mat = delta_d_fitted_peak_mat

        d_feature2_mat = d_front_mat
        d_feature2_std_mat = delta_d_front_mat
    else:
        d_wave_mat = d_front_mat
        d_wave_std_mat = delta_d_front_mat

        d_feature2_mat = d_fitted_peak_mat
        d_feature2_std_mat = delta_d_fitted_peak_mat
    #######################################



    v_min = v_min_step / 1e3  #converts form km/s to Mm/s
    v_max = v_max_step / 1e3  #converts form km/s to Mm/s

    nr_of_theta_values = d_wave_mat.shape[0]

    waves_feature_index = [[] for _ in range(nr_of_theta_values)]
    waves_time_index = [[] for _ in range(nr_of_theta_values)]
    waves_fit = [[] for _ in range(nr_of_theta_values)]
    waves_fit_cov = [[] for _ in range(nr_of_theta_values)]

    #Lists should a second feature also be fitted
    waves_fit_2 = [[] for _ in range(nr_of_theta_values)]
    waves_fit_2_cov = [[] for _ in range(nr_of_theta_values)]

    nr_of_waves_vec = np.zeros(nr_of_theta_values, dtype=np.int64)

    max_nr_peaks_const = d_wave_mat.shape[2]

    t_diff_sec = np.diff(t_sunpy_sec)

    # Set the minimum peaks per wave to either 4 (default) or 3 in the case of low cadence data
    # Warns the user if they choose to set the min_points_in_waves themselves
    #
    if (min_points_in_wave == 'default') and (np.mean(t_diff_sec) >= 60):
        min_points_in_wave = 3
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' wave_tracing_algorithm: Low cadence data detected (> 60s ), minimum peaks per wave set to 3')
    elif min_points_in_wave == 'default':
        min_points_in_wave = 4
    elif (np.mean(t_diff_sec) >= 60) and (min_points_in_wave > 3):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' wave_tracing_algorithm: Warning: Low cadence data detected (> 60s ), It is recommend to set the '
                    'set the minimum peaks per wave (min_points_in_wave) to 3')
    ##########################################################################
    #  Wave Finding Algorithm
    ##########################################################################

    # max_nr_peaks_vec[j] = np.sum(~ np.isnan(d_peak_mat[j,:,:]),None)

    t_wave_mat = np.einsum('i,j -> ij', np.arange(d_wave_mat.shape[1], dtype=np.int64),
                           np.ones(max_nr_peaks_const, dtype=np.int64))
    wave_ind_mat = np.zeros_like(d_wave_mat)  # Note: Differnt theta values start all with nr_waves = 1

    # Matrix for all the waves long enough to be tracked
    wave_ind_mat_2 = np.zeros_like(wave_ind_mat)

    for j in range(nr_of_theta_values):
        nr_waves = 0

        if max_nr_peaks_vec[j] != 0:
            for t in range(d_wave_mat.shape[1] - 1):
                for i in range(max_nr_peaks_const):
                    d_next = d_wave_mat[j, t + 1, :] - d_wave_mat[j, t, i]

                    # Sort for all allowed options
                    mask = np.zeros(max_nr_peaks_const, dtype=bool)
                    mask[(d_next > v_min * t_diff_sec[t]) & (d_next < v_max * t_diff_sec[t])] = True

                    #if j == 3:
                    #    print('hold')

                    if np.sum(mask) != 0:
                        part_of_wave_mask = wave_ind_mat[j, t + 1, mask] != 0
                        if (wave_ind_mat[j, t, i] == 0):
                            #part_of_wave_mask = wave_ind_mat[j, t + 1, mask] != 0
                            unique_wave_nr = np.unique(wave_ind_mat[j, t + 1, mask][part_of_wave_mask])
                            if np.all(~part_of_wave_mask):
                                # Creates a new wave with wave_nr +1 on both step t and t+1
                                nr_waves += 1
                                wave_ind_mat[j, t, i] = nr_waves
                                wave_ind_mat[j, t + 1, mask] = nr_waves
                            elif len(unique_wave_nr) == 1:
                                wave_ind_mat[j, t, i] = unique_wave_nr
                            else:
                                now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                                print(now + 'wave_tracing_algorithm: mulitple upstream waves claim to be origin,'
                                            'the highest wave_nr was chosen')
                                wave_ind_mat[j, t, i] = unique_wave_nr[-1] #Todo: might actually want to check the lenght of wave and decide then
                        elif np.any(part_of_wave_mask):
                            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                            print(now + 'wave_tracing_algorithm: upstream and downstream are already part of '
                                        'waves, upstream is chosen')
                            wave_ind_mat[j, t + 1, mask] = wave_ind_mat[j, t, i] #Todo: might actually want to check the lenght of wave and decide then
                        else:
                            # Adds the values at mask to the wave of the wavepoint
                            wave_ind_mat[j, t + 1, mask] = wave_ind_mat[j, t, i]

            long_enough_waves_ind_vec = []

            # Sorts all waves which are shorter than min_pints_in_wave:
            for nr_w in range(nr_waves):
                # Find the indices of the wave
                ind_vec = np.where(wave_ind_mat[j, :, :].flatten() == nr_w + 1)[0]

                # Find the corresponding time_vec indices
                time_vec = t_wave_mat.flatten()[ind_vec]

                # Check if there are enough unique time indices
                if np.unique(time_vec).size >= min_points_in_wave:
                    long_enough_waves_ind_vec.append(nr_w)

            waves_feature_index[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_time_index[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit_cov[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]

            # Lists if a second feature shall be also fited
            waves_fit_2[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit_2_cov[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]

            for w_i, nr_w in enumerate(long_enough_waves_ind_vec):
                #waves_feature[j][nr_w].append(d_wave_mat[j, :, :][wave_ind_mat[j, :, :] == nr_w + 1])  #
                #waves_time[j][nr_w].append(t_wave_mat[wave_ind_mat[j, :, :] == nr_w + 1])
                #waves_feature_std[j][nr_w].append(d_wave_std_mat[j, :, :][wave_ind_mat[j, :, :] == nr_w + 1])

                ###############################################
                #
                # Fitting the wave algorithm (added 30.12.2024)
                #
                ###############################################

                # Matrix with only waves marked deemed long enough
                wave_ind_mat_2[wave_ind_mat == nr_w + 1] = nr_w + 1

                ind_vec = np.where(wave_ind_mat[j, :, :].flatten() == nr_w + 1)[0]
                wave_p_vec = d_wave_mat[j, :, :].flatten()[ind_vec]
                time_vec = t_wave_mat.flatten()[ind_vec]
                std_vec = (d_wave_std_mat[0,j, :, :].flatten()[ind_vec] + d_wave_std_mat[1,j, :, :].flatten()[ind_vec])/2

                # Excluding nan values as there can be a detected peak but not a detected front
                not_nan = (np.isnan(wave_p_vec)) | (np.isnan(std_vec))

                t_sunpy_sec_1 = t_sunpy_sec[time_vec][~not_nan]
                wave_p_vec = wave_p_vec[~not_nan]
                std_vec = std_vec[~not_nan]

                weight_vec = 1/(std_vec)

                try_fitting_prim_feature = False
                try:

                    a, fit_cov = np.polyfit(t_sunpy_sec_1, wave_p_vec, 1,
                                            w=weight_vec,
                                            cov=True)#'unscaled')#

                    waves_feature_index[j][w_i] = ind_vec
                    waves_time_index[j][w_i] = time_vec
                    waves_fit[j][w_i] = a
                    waves_fit_cov[j][w_i] = fit_cov.flatten()

                    try_fitting_prim_feature = True
                except:
                    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                    print(
                        now + ' wave_tracing_algorithm: fitting of a wave theta_index = %.0f , wave_nr = %.0f failed' % (
                        j, nr_w))

                # Tries to fit the second feature if fitting of the primary feature was successful and if a
                # secondary feature shall be fitted
                if fit_second_feature and try_fitting_prim_feature:

                    wave_p2_vec = d_feature2_mat[j, :, :].flatten()[ind_vec]
                    std_2_vec = (d_feature2_std_mat[0,j, :, :].flatten()[ind_vec]+d_feature2_std_mat[1,j, :, :].flatten()[ind_vec])

                    #Excluding nan values as there can be a detected peak but not a detected front
                    # Also excludes any values excluded in the main fitted feature
                    not_nan2 = (np.isnan(wave_p2_vec) | np.isnan( std_2_vec)) | not_nan

                    t_sunpy_sec_2 = t_sunpy_sec[time_vec][~not_nan2]
                    wave_p2_vec = wave_p2_vec[~not_nan2]
                    std_2_vec = std_2_vec[~not_nan2]


                    weight_vec2 = 1/(std_2_vec)

                    try:

                        a, fit_cov = np.polyfit(t_sunpy_sec_2, wave_p2_vec, 1,
                                                w=weight_vec2,
                                                cov=True)#'unscaled')#

                        #waves_feature_index[j][w_i].append(ind_vec)
                        #waves_time_index[j][w_i].append(time_vec)
                        waves_fit_2[j][w_i] = a
                        waves_fit_2_cov[j][w_i] = fit_cov.flatten()
                    except:
                        waves_fit_2[j][w_i] = [np.nan]
                        waves_fit_2_cov[j][w_i] = [np.nan]
                        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                        print(now + 'wave_tracing_algorithm: fitting 2nd feature of a wave theta_index = %.0f , '
                                    'wave_nr = %.0f failed' % (j, nr_w))

            nr_of_waves_vec[j] = len(long_enough_waves_ind_vec)

    # Packing the wave values in a dict to allow for easier further us
    wave_value_dict = {}
    #name_list = ['waves_mu','waves_time','waves_std','waves_mu_std','waves_std_std','max_nr_peaks_vec','nr_of_waves_vec','distance_MM','peak_mat','d_range','mu_std_mat','std_std_mat']
    name_list = ['waves_feature_index', 'waves_time_index', 'nr_of_waves_vec', 'peak_tracked', 'max_nr_peaks_vec',
                 'nr_of_waves_vec',
                 'waves_fit', 'waves_fit_cov', 'wave_ind_mat_2', 'fit_second_feature', 'waves_fit_2', 'waves_fit_2_cov']
    for i in name_list:
        wave_value_dict[i] = eval(i)

    parameter_dict['wave tracing algorithm'] = ' '
    parameter_dict['wave tracing algorithm: v_min_step (km/s)'] = v_min_step
    parameter_dict['wave tracing algorithm: v_max_step (km/s)'] = v_max_step
    parameter_dict['wave tracing algorithm: min_points_in_wave'] = min_points_in_wave

    return wave_value_dict, parameter_dict

