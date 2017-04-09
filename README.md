# Stitch_buddy 

A Python script to stitch MicroManager multiwell mosaics and to do PIV and other analysis on time lapse microscopy images.

MicroManager has the option to generate mosaic images and to save each position of the mosaic as a separate tiff-stack.
This program, StitchBuddy, scans a directory containing OME-tiff files generated by MicroManager and stitches all mosaics it finds.
The original purpose of the program was to automatically stitch all time lapse mosaics in a multiwell plate.
It also possible to scale down the stitched mosaic, and to retain all relevant metadata, such as pixel size, time resolution, and OME-XML data.
If the image is rescaled, the imageJ-metadata is also rescaled, but currently not the OME-OXM metadata.

If the image mosaic is too large to fit in to RAM, there is an option to stitch the file frame-by-frame on disk but this is very, very slow.

All tiff file reading/writing and metadata is handled by  [tifffile.py](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)

#Cvv_analysis
Script to perform PIV analysis (with openPIV), and to calculate the alignment index, mean square velocity, Root mean square velocity, the instantaneous order parameter, and the vectorial angular correlation length from the PIV data.

The alignment index is defined as the dot product of a velocity vector (vextor_x) and the average velocity vector (vector_0) for the frame, divided by the product of their magnitudes. Like so: alignment_index = (vector_x dot vector_0)/(|vector_x||vector_0|).

Mean square velocity (msv, $\langle|\textbf{v}(t)|^2\rangle$ ), Root mean square velocity ($\nu_{r.m.s}$), the instantaneous order parameter ($\psi$), and finally the vectorial velocity correlation function ($C_{VV}(r)$)