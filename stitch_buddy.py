from __future__ import print_function, division, absolute_import
import re
import os
import numpy as np
import tiffile as tiffile
import time
from fractions import Fraction
import warnings

def filenamesToDict(indir, wellNameDict=None):

	"""
	Transforms the (ome).tif stack-files in a directory into a dictionary.

	Filenames must conform to the following general pattern:
	..._wellID-xxx_threeDigitRowNumber_threeDigitColNumber...

	:param
	indir: path to directory with files from a multiwell mosaic experiment
	wellNameDict: Dictionary of names to give the wells from the wellID-number

	property_dict = 'nrows': (int) No. of rows in well
					'ncols': (int) No. of columns in well
					'nChans': (int) No. of channels in image
					'xpix': (int) No. pixels in X-dimension of images
					'ypix': (int) No. pixels in Y-dimension of images
					'frame_interval': (float) No. of time units between frames
					'timeUnit': (str) time unit
					'pixres': (float) size of pixels in resolution units
					'resUnit': (str) spatial resolution unit
					'pixelType':(str) pixeldepth of image
					'positions': position_dict gives file at position (row,col)
					'files':(list) names of files in well
					'isConcat':(bool) If the sequence is split in to multiple files
					'OME-XML':None <- Not implemented yet

	position_dict = {(row, col):filename(s)}



	:return: Dictionary with wellID:property_dict
	"""
	# Ingore non-.tif files in indir
	filenames = [fname for fname in os.listdir(indir) if ".tif" in fname]
	filenames.sort()

	wellDict = {}

	isConcat = False

	#Regex used to flexibly identify wells, rows, columns, and split files from filenames

	#Matches any number of digits preceded by "MMStack_" and followed by "-"
	well_regex = re.compile("(?<=MMStack_)\d+(?=-)")

	#Matches three digits preceded by three digits and "_"
	column_regex = re.compile("(?<=\d{3}_)\d{3}")

	#Matches three digits preceded by three digits and "_"
	row_regex = re.compile("(?<=_)\d{3}(?=_)")

	#Matches a digit preceded by three digits and "_", followed by ".ome"
	concat_regex = re.compile("(?<=\d{3}_\d{3}_)\d(?=\.ome)")

	first_file = os.path.join(indir, filenames[0])
	print("Opening the first file: \"%s\" to read its MicroManager metadata..." % first_file)


	with tiffile.TiffFile(first_file) as tif:

		frames = len(tif.pages)
		page = tif[0]
		print("There are %s frames and their shape is %s" % (frames, page.shape) )

		pixres=page.tags['x_resolution'].value #Assumes equal x/y resolution
		resoloution_unit = page.tags['resolution_unit'].value
		resoloution_unit = {1: 'none', 2: 'inch', 3: 'centimeter'}[resoloution_unit]
		pixelDepth = page.tags['bits_per_sample'].value
		omexml = page.tags['image_description'].value

		try:
			meta_data = tif.micromanager_metadata
			frame_interval = meta_data['summary']['WaitInterval']
			ypix, xpix = meta_data['summary']['Height'], meta_data['summary']['Width']
			nChannels = meta_data['summary']['Channels']
			nSlices = meta_data['summary']['Slices']
			nTimepoints = frames / (nChannels * nSlices)

		except:
			warnings.warn("Metadata read error!")
			print("Something is wrong with the MicroManager metadata, replacing all values with defaults, this means"
				  " that SCALING IS NOT CORRECT and the image stack is flattened over channels and slices!")
			frames = page.shape[0]
			meta_data = None
			frame_interval = 1
			ypix, xpix = page.shape[1], page.shape[2]
			nChannels = 1
			nSlices = 1
			nTimepoints = frames

	for f in filenames:
		#Extract positioning information from filename with regex
		wellID = int(well_regex.search(f).group())


		if wellNameDict != None:
			wellID = wellNameDict[wellID]


		rowNumber = int(row_regex.search(f).group())
		columnNumber = int(column_regex.search(f).group())
		concat = concat_regex.search(f)
		if concat != None:
			isConcat = True

		#If there is no key for wellID in wellDict -> create a dict of properties
		if wellDict.get(wellID) == None:
			wellDict[wellID] = {'nrows':1,
								'ncols':1,
								'nChannels':int(nChannels),
								'nSlices': int(nSlices),
								'xpix':int(xpix),
								'ypix':int(ypix),
								'nTimepoints':int(nTimepoints),
								'frame_interval':(frame_interval/1000.0),
								'timeunit':'s',
								'pixel_resolution':pixres, #resolution stored as rational in tif tag
								'resoloution_unit': resoloution_unit,
								'pixelDepth':pixelDepth,
								'positions':{},
								'files':[],
								'isConcat':isConcat,
								'OME-XML':omexml
								}

		#Populate Properties
		wellDict[wellID]['nrows'] = max(rowNumber+1, wellDict[wellID]['nrows'])
		wellDict[wellID]['ncols'] = max(columnNumber+1, wellDict[wellID]['ncols'])

		#List of filenames for the well
		wellDict[wellID]['files'].append(f)

		#Dict with (row, column):(list) filename(s)
		if wellDict[wellID]['positions'].get((rowNumber, columnNumber)) == None:
			wellDict[wellID]['positions'][(rowNumber, columnNumber)]=[f]
		else:
			wellDict[wellID]['positions'][(rowNumber, columnNumber)].append(f)
			wellDict[wellID]['positions'][(rowNumber, columnNumber)].sort()
			wellDict[wellID]['isConcat'] = isConcat

	print("wellDict created, proceed to choosing output directory and renaming wells.")
	return wellDict

def bin_ndarray(ndarray, new_shape, operation='mean'):
	"""
	Bins an ndarray in all axes based on the target shape, by summing,
		averaging, or returning min/max pixel value

	Number of output dimensions must match number of input dimensions and
		new axes must divide old ones.

	Example
	-------
	>>> m = np.arange(0,100,1).reshape((10,10))
	>>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
	>>> print(n)

	[[ 22  30  38  46  54]
	 [102 110 118 126 134]
	 [182 190 198 206 214]
	 [262 270 278 286 294]
	 [342 350 358 366 374]]

	"""
	operation = operation.lower()
	if not operation in ['sum', 'mean', 'max', 'min']:
		raise ValueError("Operation not supported.")
	if ndarray.ndim != len(new_shape):
		raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
														   new_shape))
	compression_pairs = [(d, c//d) for d,c in zip(new_shape,
												  ndarray.shape)]
	flattened = [int(l) for p in compression_pairs for l in p]
	ndarray = ndarray.reshape(flattened)
	for i in range(len(new_shape)):
		op = getattr(ndarray, operation)
		ndarray = op(-1*(i+1))
	return ndarray

def stitchWellsOnDisk(wellDict, inputDir, outputDir, resizeTo=None):
	"""
	Avoids loading constituent files in to RAM. Stitches and rescales frame-by frame instead.
	Args:
		wellDict: (dict) wellID:filename
		inputDir: str or os.path
		outputDir: str or os.path
		resizeTo: Factor to rezie to, mus be a factor of 2

	Returns:

	"""


	tStart=time.time()

	for well in wellDict.keys():
		t0=time.time()
		print("Starting on wellID:", well)
		ncols = wellDict[well]['ncols']
		nrows = wellDict[well]['nrows']
		nChans = wellDict[well]['nChannels']
		nTimepoints = wellDict[well]['nTimepoints']
		nSlices = wellDict[well]['nSlices']
		frame_interval = wellDict[well]['frame_interval']
		time_unit  = wellDict[well]['timeunit']
		pixelDepthDict = {8: "uint8", 16:"uint16", 32:"float32"}
		pixType = pixelDepthDict[wellDict[well]['pixelDepth']]
		xpix = wellDict[well]['xpix']
		ypix = wellDict[well]['ypix']
		pixel_resolution = wellDict[well]['pixel_resolution']

		outWidth = xpix*ncols
		outHeight = ypix*nrows

		if resizeTo != None:

			print("Resizing outArray...")
			old_resolution = pixel_resolution[0]/float(pixel_resolution[1])
			print("old resolution; {}, rational: {}".format(old_resolution,
															pixel_resolution))
			new_resolution = old_resolution*resizeTo
			rational_new_resolution = Fraction(new_resolution).limit_denominator()

			pixel_resolution = (rational_new_resolution.numerator,
								rational_new_resolution.denominator)

			print("new resolution; {}, rational: {}".format(new_resolution,
															pixel_resolution))
			outArray = np.empty(
				(nTimepoints, nSlices, nChans, outHeight*resizeTo, outWidth*resizeTo),
				dtype=pixType)

		else:
			outArray = np.empty(
				(nTimepoints, nSlices, nChans, outHeight, outWidth),
				dtype=pixType)

		full_frame_buffer = np.empty((1, nSlices, nChans, outHeight, outWidth),
				dtype=pixType)

		print("output well array shape: {}, full_frame_array shape; {}".format(outArray.shape, full_frame_buffer.shape) )


		for frame in range(nTimepoints):
			t1 = time.time()
			print("Working on frame: {}".format(frame))
			for row in range(nrows):
				for col in range(ncols):
					#get name of file at this row, col position
					loadme = os.path.join(inputDir,
										  wellDict[well]['positions'][(row, col)][0]
										  )
					#X/Y coordinates for insertion of subframe
					startX = (ncols-col-1)*xpix
					startY = row*ypix

					#is_ome is set to False so that all .asarray calls will return the same shape
					with tiffile.TiffFile(loadme, is_ome = False) as tif:
						#only get current frame with channels and slices as an array
						if nSlices == 1:
							slice_to_load = slice(frame * nChans,
												  (frame + 1) * nChans)
						else:
							slice_to_load = slice(frame*(nChans+nSlices),
												  (frame+1)*(nChans+nSlices))

						inArray = tif.asarray(key=slice_to_load)

						try:
							full_frame_buffer[:,:,:, startY:(startY + ypix), startX:(startX + xpix)] = inArray
						except:
							inArray = np.reshape(inArray, (1, nSlices, nChans, xpix, ypix))
							#print("Input array reshaped to: {}".format(inArray.shape))
							full_frame_buffer[:,:,:, startY:(startY + ypix), startX:(startX + xpix)] = inArray
						print("Frame: {}, Row: {}, Col: {} loaded from file: {} array shape: {}".format(
							frame, row, col, tif.filename, inArray.shape))


			if resizeTo != None:

				resized_frame_buffer = bin_ndarray(full_frame_buffer, (1, nSlices, nChans, outHeight*resizeTo, outWidth*resizeTo))
				print("full_fame_buffer resized to {}".format(resized_frame_buffer.shape))

				outArray[frame,:,:,:,:] = resized_frame_buffer

			else:
				outArray[frame,:,:,:,:] = full_frame_buffer

			print("Frame, done in {} s. Elapsed time for well: {} min,  Since start: {} min".format(round((time.time() - t1), 1),
																								round((time.time() - t0)/60),
																								round((time.time() - tStart)/60)))

		saveme=os.path.join(outputDir, str(well)+"_stitched.tif")
		bigTiffFlag = outArray.size * outArray.dtype.itemsize > 2000 * 2 ** 20
		print("bigTillFlag set to:", bigTiffFlag, "Saving output...(may take a while)")

		metadata = {"zStack" : bool(1-nSlices),
					"unit":"um",
					"tunit": time_unit,
					"finterval" : frame_interval
					}
		imageJresolution = (pixel_resolution[0]/pixel_resolution[1])/10000.0

		save_data = {"bigtiff" : bigTiffFlag,
					"imagej" : True,
					"resolution" : (imageJresolution, imageJresolution, None),
					"metadata" : metadata
					}

		tiffile.imsave(saveme, outArray, **save_data)

		print("Done with wellID: ", well, "in ", round(time.time() - t0, 2), " s")
	print("All done, it took ", round(time.time() - tStart, 2), " s in total!")

def stitchWellsInRAM(wellDict, inputDir, outputDir, resizeTo=None):

	tStart=time.time()

	for well in wellDict.keys():
		t0=time.time()
		print("Starting on wellID:", well)
		ncols = wellDict[well]['ncols']
		nrows = wellDict[well]['nrows']
		nChans, nTimepoints = wellDict[well]['nChannels'], wellDict[well]['nTimepoints']
		nSlices = wellDict[well]['nSlices']
		frame_interval, time_unit  = wellDict[well]['frame_interval'], wellDict[well]['timeunit']

		pixelDepthDict = {8: "uint8", 16:"uint16", 32:"float32"}
		pixType = pixelDepthDict[wellDict[well]['pixelDepth']]


		xpix, ypix, pixel_resolution = wellDict[well]['xpix'], wellDict[well]['ypix'], wellDict[well]['pixel_resolution']
		outWidth = xpix*ncols
		outHeight = ypix*nrows

		outArray = np.empty((nTimepoints, nSlices, nChans, outHeight, outWidth), dtype=pixType)

		print("well array shape:", outArray.shape)

		for r in range(nrows):
			for c in range(ncols):
				t1 = time.time()
				startX = (ncols-c-1)*xpix
				startY = r*ypix
				loadme = os.path.join(inputDir, wellDict[well]['positions'][(r,c)][0])
				print("Working on file: ", str(loadme))

				with tiffile.TiffFile(loadme) as tif:
					inArray = tif.asarray()
					print("File loaded as array, shape: ", inArray.shape, "loadtime: ", round(time.time() - t1), " s")

				try:
					outArray[:,:,:, startY:(startY+ypix), startX:(startX+xpix)] = inArray
				except:
					inArray = np.reshape(inArray, (nTimepoints, nSlices, nChans, xpix, ypix))
					print("Input array reshaped to: ", inArray.shape)
					outArray[:,:,:, startY:(startY + ypix), startX:(startX + xpix)] = inArray

				print("Input array appended to OutArray. Elapsed time for well: ", round(time.time() - t0))

		saveme=os.path.join(outputDir, str(well)+"_stitched.tif")

		if resizeTo != None:
			print("Resizing outArray...")
			old_resolution = pixel_resolution[0]/float(pixel_resolution[1])
			print("Original pixel resolution was: %s / %s = %s px/resolution unit" % (pixel_resolution[0], pixel_resolution[1], old_resolution))
			new_resolution = old_resolution*float(resizeTo)
			rational_new_resolution = Fraction(new_resolution).limit_denominator()

			pixel_resolution = (rational_new_resolution.numerator,
								rational_new_resolution.denominator)

			print("New pixel resolution is: %s / %s = %s px/resolution unit" % (pixel_resolution[0], pixel_resolution[1], new_resolution))

			t=time.time()
			outArray = bin_ndarray(outArray, ((nTimepoints, nSlices, nChans,
											   outHeight*resizeTo,
											   outWidth*resizeTo))).astype("uint16")
			print("Done in %.2f s with resizing output!" % (round(time.time()-t)))

		bigTiffFlag = outArray.size * outArray.dtype.itemsize > 2000 * 2 ** 20
		print("bigTillFlag set to:", bigTiffFlag, "Saving output...(may take a while)")

		metadata = {"zStack" : bool(1-nSlices),
					"unit":"um",
					"tunit": time_unit,
					"finterval" : frame_interval
					}

		imageJresolution = (pixel_resolution[0]/pixel_resolution[1])/10000.0

		save_data = {"bigtiff" : bigTiffFlag,
					"imagej" : True,
					"resolution" : (imageJresolution, imageJresolution, None),
					"metadata" : metadata
					}

		tiffile.imsave(saveme, outArray, **save_data)

		print("Done with wellID: ", well, "in ", round(time.time()-t0,2), " s")

	print("All done, it took ", round(time.time() - tStart, 2), " s in total!")







