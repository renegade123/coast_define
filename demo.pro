pro demo
  myshape = OBJ_NEW('IDLffShape',"d:\example\pre_line.shp")
  ;  ENVI,/restore_base_save_files
  ;  ENVI_BATCH_INIT
  ;  ENVI_OPEN_FILE,infile,r_fid = fid
  ;  ENVI_FILE_QUERY, fid, ns = ns, nb = nb, nl = nl, dims = dims,BNAMES = BNAMES
  ;  ; Get the number of entities so we can parse through them
  ;  PRINT, myshape
  myshape->getproperty,n_entities=n_ent,Attribute_info=attr_info,n_attributes=n_attr,Entity_type=ent_type
  ent=myshape->getentity(0) ; get no. i entity
  bounds=ent.BOUNDS ;read bounds
  n_vert=ent.N_VERTICES ;only for polyline and polygon
  vert=*(ent.VERTICES) ;only for polyline and polygon
  obj = obj_new("coastline",'C:\Users\name\IDLWorkspace83\coastline__define\subset_VV.tif',1)
  ;obj->getline
  ;obj->pre_line,vert=vert
  ;obj->pre_contour,vert=vert
  ;obj->pre_line,vert=vert
  ;if Obj_Valid(obj) then Obj_Destroy, obj
  ;obj = obj_new("coastline",'E:\IDLWorkspace83\newcoastline\cc.JPG',2)
  obj->getline
end