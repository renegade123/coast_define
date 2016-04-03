;+
; 《IDL语言程序设计》
; --数据可视化与ENVI二次开发
;
; 示例程序
;
; 作者: 董彦卿
;
; 联系方式：sdlcdyq@sina.com
;
;-
;todo:根据两个点返回矩形的四个点坐标
FUNCTION coastline::CalRectPoints,ulPos,drPos

  self.OWINDOW.GETPROPERTY, dimensions = winDims,graphics_tree = oView
  oView.GETPROPERTY, viewPlane_Rect = viewRect
  lLoc = viewRect[0:1]+[ulPos[0],ulPos[1]]*viewRect[2:3]/winDims
  rLoc = viewRect[0:1]+[drPos[0],drPos[1]]*viewRect[2:3]/winDims
  IF ABS(rLoc[0]-lLoc[0]) EQ 0 THEN rLoc[0]= lLoc[0]+1
  IF ABS(rLoc[1]-lLoc[1]) EQ 0 THEN rLoc[1]= lLoc[1]+1
  RETURN,[[lLoc],[lLoc[0],rLoc[1]],$
    [rLoc],[rLoc[0],lLoc[1]]]
END


;todo:销毁析构
PRO coastline::CLEANUP
  IF  PTR_VALID(self.ORIDATA) THEN PTR_FREE,self.ORIDATA
  IF  PTR_VALID(self.conData) THEN PTR_FREE,self.conData
  IF  PTR_VALID(self.preConData) THEN PTR_FREE,self.preConData
  ;IF OBJ_VALID(self.OWINDOW) THEN OBJ_DESTROY, self.OWINDOW
  IF OBJ_VALID(self.OIMAGE) THEN OBJ_DESTROY, self.OIMAGE
  IF OBJ_VALID(self.ORECT) THEN OBJ_DESTROY, self.ORECT
  IF OBJ_VALID(self.oContour) THEN OBJ_DESTROY, self.oContour
  IF OBJ_VALID(self.oPreContour) THEN OBJ_DESTROY, self.oPreContour
  IF OBJ_VALID(self.oPreLine) THEN OBJ_DESTROY, self.oPreLine
  IF OBJ_VALID(self.INITLINE) THEN OBJ_DESTROY, self.INITLINE
END

;todo:修改组件大小是
PRO coastline::ChangeDrawSize,width,height
  IF N_ELEMENTS(width) THEN BEGIN
    self.OWINDOW.GETPROPERTY, graphics_tree = oView
    oView.GetProperty,ViewPlane_Rect = viewP,dimensions = dims
    oriWL = viewP[2:3]
    viewP[2:3] =viewP[2:3]*[width,height]/dims
    viewP[0:1]+=(oriWL-viewP[2:3])/2

    oView.SETPROPERTY,dimension = [width,height],viewPlane_Rect= viewP
    self.OWINDOW.Draw
  ENDIF
END


;todo:NeumannBoundCond
FUNCTION coastline::NeumannBoundCond,f
  size = SIZE(f);
  nrow=size[1]
  ncol=size[2]
  g = f;
  ;g[[0,nrow-1],[0,ncol-1]] = g([2,nrow-3],[2,ncol-3]);
  g[0,0]=g[2,2]
  g[0,ncol-1]=g[2,ncol-3]
  g[nrow-1,0]=g[nrow-3,2]
  g[nrow-1,ncol-1]=g[nrow-3,ncol-3]
  ;g[[1,nrow-1],1:ncol-2] = g[[2,nrow-3],1:ncol-2]
  g[0,1:ncol-2] = g[2,1:ncol-2]
  g[nrow-1,1:ncol-2] = g[nrow-3,1:ncol-2]
  ;g[1:nrow-2,[0,ncol-1]] = g[1:nrow-2,[2, ncol-3]]
  g[1:nrow-2,0] = g[1:nrow-2,2]
  g[1:nrow-2,ncol-1] = g[1:nrow-2, ncol-3]
  RETURN,g
END


;todo:div
FUNCTION coastline::div,nx,ny
  nxGrad=self.gradient(nx,/vector)
  nxx=nxGrad[*,*,0]
  junk=nxGrad[*,*,1]
  nyGrad=self.gradient(ny,/vector)
  junk=nyGrad[*,*,0]
  nyy=nyGrad[*,*,1]
  f=nxx+nyy
  RETURN,f
END


;todo Dirac
FUNCTION coastline::Dirac,x,sigma
  f=(0.5/sigma)*(1+COS(!PI*x/sigma));
  m = (x LE sigma)
  n = (x GE -sigma)
  b =  LOGICAL_AND(m,n)
  res = f*b;
  RETURN,res
END


;TODO del2
FUNCTION coastline::del2,f
  sz = SIZE( f )
  nx=sz[1] & ny=sz[2]
  L=DINDGEN(nx,ny)
  FOR i=1,nx-2 DO BEGIN
    FOR j=1,ny-2 DO BEGIN
      L[i,j]=(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])/4-f[i,j]
    ENDFOR
  ENDFOR
  L[0,*]=0
  L[*,0]=0
  L[nx-1,*]=0
  L[*,ny-1]=0
  ;    L[0,0]=(f[2,0]+f[0,2]+2*f[0,0])/4-(f[1,0]+f[0,1])/2;左上角
  ;    L[nx-1,0]=(f[nx-3,0]+f[nx-1,2]+2*f[nx-1,0])/4-(f[nx-2,0]+f[nx-1,1])/2;左下角
  ;    L[0,ny-1]=(f[0,ny-3]+f[2,ny-1]+2*f[0,ny-1])/4-(f[0,ny-2]+f[1,ny-1])/2;右上角
  ;    L[nx-1,ny-1]=(f[nx-3,ny-1]+f[nx-1,ny-3]+2*f[nx-1,ny-1])/4-(f[nx-2,ny-1]+f[nx-1,ny-2])/2;右下角
  ;    for j=1,ny-2 do begin
  ;        L[0,j]=(f[0,j+1]+f[0,j-1]+f[2,j]+f[0,j])/4-(f[0,j]+f[1,j])/2
  ;    endfor
  ;    FOR j=1,ny-2 DO BEGIN
  ;      L[nx-1,j]=(f[nx-1,j+1]+f[nx-1,j-1]+f[nx-3,j]+f[nx-1,j])/4-(f[nx-1,j]+f[nx-2,j])/2
  ;    ENDFOR
  ;    FOR j=1,nx-2 DO BEGIN
  ;      L[j,0]=(f[j+1,0]+f[j-1,0]+f[j,2]+f[j,0])/4-(f[j,0]+f[j,1])/2
  ;    ENDFOR
  ;    FOR j=1,nx-2 DO BEGIN
  ;      L[j,ny-1]=(f[j+1,ny-1]+f[j-1,ny-1]+f[j,ny-3]+f[j,ny-1])/4-(f[j,ny-1]+f[j,ny-2])/2
  ;    ENDFOR
  RETURN,L
END


;toDO 梯度
FUNCTION coastline::gradient, image, vector=vector, norm=norm,  $
  forward=forward,central=central,$
  scale=scale
  ;+
  ; NAME:
  ; gradient
  ; PURPOSE:
  ; Compute the gradient vector at every pixel of an image using
  ; neighbor pixels. Default is to return the Euclidean norm of gradient
  ; (2-D array), or specify keywords for other options.
  ; CALLING:
  ; gradim = gradient( image )
  ; INPUTS:
  ; image = 2-D array.
  ; KEYWORDS:
  ; scale=scale,or [xscale,yscale]; scale means [scale,scale] for a single value, scale=[xscale,yscale]
  ; /vector then 2 images are returned (3-D array) with d/dx and d/dy.
  ; /forward then x & y-partial derivatives computed as forward difference.
  ;       /central, or default cenral difference
  ; /norm then magnitude of gradient is computed as norm.
  ; OUTPUTS:
  ; Function returns gradient norm (2-D array) or gradient vector (3-D).
  ; HISTORY:
  ;                   10-Feb-2011, by Ding Yuan( Ding.Yuan@warwick.ac.uk)
  ;-
  ;if ~exist(image) then on_error
  sz = SIZE( image )
  IF (sz[0] NE 2) THEN MESSAGE,"Please input an 2D image (array)"
  nx=sz[1] & ny=sz[2]
  option=0
  ;if exist(forward) then option=1
  CASE option OF
    0: BEGIN
      didx = ( SHIFT( image, -1,  0 ) - SHIFT( image, 1, 0 ) ) * 0.5
      didx[0,*] = image[1,*] - image[0,*]
      didy = ( SHIFT( image,  0, -1 ) - SHIFT( image, 0, 1 ) ) * 0.5
      didy[*,0] = image[*,1] - image[*,0]
    END
    1:  BEGIN
      didx = SHIFT( image, -1,  0 ) - image
      didy = SHIFT( image,  0, -1 ) - image
    END
    ELSE: ON_ERROR,2
  ENDCASE
  didx[nx-1,*] = image[nx-1,*] - image[nx-2,*]
  didy[*,ny-1] = image[*,ny-1] - image[*,ny-2]
  xscale=1 & yscale=1
  CASE N_ELEMENTS(scale) OF
    0: BEGIN
      xscale=1 & yscale=1
    END
    1:BEGIN
    xscale=scale & yscale=scale
  END
  2:BEGIN
  xscale=scale[0] & yscale=scale[2]
END
ELSE: MESSAGE,'scale should be a scalar or 2 element vector'
ENDCASE
didx = didx * xscale
didy = didy * yscale
IF KEYWORD_SET(vector ) THEN  RETURN, [ [[didx]], [[didy]] ] ELSE RETURN , SQRT( didx*didx + didy*didy )
END


;TODO边界
FUNCTION coastline::drlse_edge,phi, g, lambda,mu, alfa, epsilon, timestep, iter, potentialFunction
  phi=phi
  grad_g=self.gradient(g,/vector)
  vx=grad_g[*,*,0]
  vy=grad_g[*,*,1]
  FOR k=1,iter DO BEGIN
    phi=self.NeumannBoundCond(phi)
    ;todo:求梯度phi_x,phi_y
    grad_phi=self.gradient(phi,/vector)
    phi_x=grad_phi[*,*,0]
    phi_y=grad_phi[*,*,1]
    s=SQRT(phi_x^2 + phi_y^2)
    smallNumber=1e-10
    Nx=phi_x/(s+smallNumber)
    Ny=phi_y/(s+smallNumber)
    curvature=self.div(Nx,Ny);
    IF STRCMP(potentialFunction,'single-well') THEN BEGIN
      distRegTerm = 4*self.del2(phi)-curvature
    ENDIF ELSE BEGIN
      IF  STRCMP(potentialFunction,'double-well') THEN distRegTerm=self.distReg_p2(phi) ELSE PRINT,'Error: Wrong choice'
    END
    diracPhi=self.Dirac(phi,epsilon)
    areaTerm=diracPhi*g
    edgeTerm=diracPhi*(vx*Nx+vy*Ny) + diracPhi*g*curvature
    phi=phi + timestep*(mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm)
  ENDFOR
  RETURN,phi
END


;TODO distReg_p2
FUNCTION coastline::distReg_p2,phi
  phiGrad=self.gradient(phi,/vector)
  phi_x=phiGrad[*,*,0]
  phi_y=phiGrad[*,*,1]
  s=SQRT(phi_x^2 + phi_y^2);
  sz = SIZE( phi )
  nx=sz[1] & ny=sz[2]

  m = (s GE 0)
  n = (s LE 1)
  a = LOGICAL_AND(m,n)
  b=(s GT 1)

  ps=a*SIN(2*!pi*s)/(2*!pi)+b*(s-1);  compute first order derivative of the double-well potential p2 in eqaution (16)
  dps=((ps NE 0)*ps+(ps EQ 0))/((s NE 0)*s+(s EQ 0));  compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
  f = self.div(dps*phi_x - phi_x, dps*phi_y - phi_y) + 4*self.del2(phi)
  RETURN,f
END


;TODO 提取海岸线
PRO coastline::getline,inifile
  ;originImg=READ_IMAGE('E:\IDLWorkspace83\newcoastline\12.png')
  ;originImg=READ_IMAGE('E:\IDLWorkspace83\newcoastline\cc.JPG')
  ;originImg=READ_tiff(self.infile)
  originImg=READ_TIFF('C:\Users\name\IDLWorkspace83\coastline\subset_VV.tif',R, G, B,GEOTIFF=GeoKeys,INTERLEAVE = 0)
  ;Img=TRANSPOSE(ROTATE(DOUBLE(REFORM(originImg[0,2000:2399,2000:2399])),1))
  ;Img=TRANSPOSE(ROTATE(DOUBLE(REFORM(originImg[1,*,*])),1))
  Img=DOUBLE(originImg)
  HELP,IMG
  ;plot, transpose(img,[0,2,1])
  timestep=5;   time step
  mu=0.2/timestep;coefficient of the distance regularization term R(phi)
  iter_inner=5
  ;iter_outer=150
  iter_outer=300
  lambda=5;coefficient of the weighted length term L(phi)
  ;alfa=1.5;coefficient of the weighted area term A(phi)
  alfa=5000;
  epsilon=1.5;papramater that specifies the width of the DiracDelta function
  sigma=1.5;scale parameter in Gaussian kernel
  ;  Gauss=GAUSSIAN_FUNCTION([sigma,sigma],/double,width=15);gauss核
  ;  Img_smooth=convol(Img,gauss,/CENTER, /EDGE_TRUNCATE);卷积
  Img_smooth = GAUSS_SMOOTH(Img , sigma,/EDGE_ZERO); /EDGE_ZERO gauss平滑
  iGrad=self.gradient(Img_smooth,/vector);
  Ix=iGrad[*,*,0]
  Iy=iGrad[*,*,1]
  f=Ix^2+Iy^2;
  g=1/(1+f);
  c0=2
  imgsize=SIZE(Img)
  initialLSF=c0*MAKE_ARRAY(imgsize[1],imgsize[2],/double,VALUE=1)
  ;initialLSF[0:500,0:370]=-c0
  ;initialLSF[0:289,0:279]=-c0
  ;initialLSF[0:399,40:399]=-c0
  initialLSF[2:1478,2:1559]=-c0
  phi=initialLSF


  ;显示初始level——set
  ;im = IMAGE(originImg[*,2000:2399,2000:2399], RGB_TABLE=13, TITLE='Coastline')
  TileData = BYTSCL(originImg)
  ;self.ORIDATA = PTR_NEW(TileData,/no_Copy)
  self.ORIDATA = PTR_NEW(TileData,/no_Copy)
  ;self.CONDATA = PTR_NEW(TRANSPOSE(ROTATE(phi,1)),/no_Copy)
  self.CONDATA = PTR_NEW(initialLSF,/no_Copy)
  idata = *(self.ORIDATA)
  cdata = *(self.CONDATA)
  self.OIMAGE.SETPROPERTY, data = idata
  self.oContour.SETPROPERTY, hide =0,data = cdata
  self.OWINDOW.draw
  ;im = IMAGE(originImg, RGB_TABLE=13, TITLE='Coastline',/OVERPLOT)
  ;c=CONTOUR(TRANSPOSE(ROTATE(phi,1)), C_LINESTYLE=0,c_label_show=0,COLOR=[0,255,0] ,c_value=[0,0] , /OVERPLOT)
  potential=2;
  IF potential EQ 1 THEN BEGIN
    potentialFunction = 'single-well';use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
  ENDIF ELSE BEGIN
    IF potential EQ 2 THEN potentialFunction = 'double-well' ELSE potentialFunction = 'double-well'
  END
  FOR n=1,iter_outer DO BEGIN
    phi=self.drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
    IF (n MOD 2) EQ 0 THEN BEGIN
      ;      c.erase
      ;      im = IMAGE(originImg[*,2000:2399,2000:2399], RGB_TABLE=13, TITLE='Coastline',/OVERPLOT)
      ;      ;im = IMAGE(originImg, RGB_TABLE=13, TITLE='Coastline',/OVERPLOT)
      ;      c = CONTOUR(TRANSPOSE(ROTATE(phi,1)), C_LINESTYLE=0,c_label_show=0,COLOR=[0,255,0] ,c_value=[0,0] ,/OVERPLOT)
      ;self.CONDATA = PTR_NEW(TRANSPOSE(ROTATE(phi,1)),/no_Copy)
     tempphi = phi
      self.CONDATA = PTR_NEW(tempphi,/no_Copy)
      cdata = *(self.CONDATA)
      self.OCONTOUR.SETPROPERTY, hide =0,data = cdata
      self.OWINDOW.draw
    ENDIF
  ENDFOR
  alfa=0;
  iter_refine = 10;
  phi = self.drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
  ;  c.erase
  ;  im = IMAGE(originImg[*,2000:2399,2000:2399], RGB_TABLE=13, TITLE='Coastline',/OVERPLOT)
  ;  ;im = IMAGE(originImg, RGB_TABLE=13, TITLE='Coastline',/OVERPLOT)
  ;  c = CONTOUR(TRANSPOSE(ROTATE(phi,1)), C_LINESTYLE=0,c_label_show=0,COLOR=[0,255,0] ,c_value=[0,0] ,/OVERPLOT)
  ;self.CONDATA = PTR_NEW(TRANSPOSE(ROTATE(phi,1)),/no_Copy)
  self.CONDATA = PTR_NEW(phi,/no_Copy)
  cdata = *(self.CONDATA)
  self.OCONTOUR.SETPROPERTY, hide =0,data = cdata
  self.OWINDOW.draw
  B = WHERE(phi GT 0, count, COMPLEMENT=B_C, NCOMPLEMENT=count_c)
  phi[b] = -1
  phi[b_c] = 1
  WRITE_TIFF,"d:\test.tif",phi
END


;;矢量数据读入
;function read_shape
;  ;COMPILE_OPT idl2
;  ; Open the states Shapefile in the examples directory
;  ;  myshape = OBJ_NEW('IDLffShape',FILEPATH( "continents.shp", SUBDIRECTORY=["resource","maps","shape"] ))
;  ;  infile = FILEPATH( "day.jpg", SUBDIRECTORY=["examples","data"] )
;  myshape = OBJ_NEW('IDLffShape',"d:\example\pre_line.shp")
;  ;  ENVI,/restore_base_save_files
;  ;  ENVI_BATCH_INIT
;  ;  ENVI_OPEN_FILE,infile,r_fid = fid
;  ;  ENVI_FILE_QUERY, fid, ns = ns, nb = nb, nl = nl, dims = dims,BNAMES = BNAMES
;  ;  ; Get the number of entities so we can parse through them
;  ;  PRINT, myshape
;  myshape->getproperty,n_entities=n_ent,Attribute_info=attr_info,n_attributes=n_attr,Entity_type=ent_type
;  ent=myshape->getentity(0) ; get no. i entity
;  bounds=ent.BOUNDS ;read bounds
;  n_vert=ent.N_VERTICES ;only for polyline and polygon
;  vert=*(ent.VERTICES) ;only for polyline and polygon
;  plot,vert
;end


;TODO 原始海岸线
PRO coastline::pre_line,vert=vert
  ;graph=polygon(vert[0,*],vert[0,*],'-r2')
  ;ENVI_CONVERT_FILE_COORDINATES,fid,xmap,ymap,vert[0,*],vert[1,*]
  ;map_coord = [xmap,ymap]
  ;plot,vert
  self.OPRELINE.SETPROPERTY, hide =0,data = vert
  self.OWINDOW.draw
END


;原始矢量海岸线
PRO coastline::pre_contour,vert=vert
  ;graph=polygon(vert[0,*],vert[0,*],'-r2')
  ;ENVI_CONVERT_FILE_COORDINATES,fid,xmap,ymap,vert[0,*],vert[1,*]
  ;map_coord = [xmap,ymap]
  ;plot,vert
  mask_obj = OBJ_NEW("IDLanROI",vert[0,*],vert[1,*])
  mask_arr = mask_obj->ComputeMask(DIMENSIONS=[297,289],INITIALIZE=0,MASK_RULE=2)
  mask_arr = DOUBLE(mask_arr)
  self.preConData = PTR_NEW(TRANSPOSE(ROTATE(mask_arr,1)),/no_Copy)
  cdata = *(self.preConData)
  self.opreContour.SETPROPERTY, hide =0,data = cdata
  self.OWINDOW.draw
END
;TODO 初始海岸线选择
PRO coastline::line_init
  WHILE (!mouse.button NE 4) DO BEGIN
    CURSOR, x1, y1, /norm, /down
    PLOTS,[x,x1], [y,y1], /normal
    x = x1 & y = y1
  ENDWHILE
END


;TODO 栅格转矢量
PRO coastline::raster_to_vector
  COMPILE_OPT idl2
  ;ENVI调用初始化
  ENVI,/restore_base_save_files
  ENVI_BATCH_INIT
  ;打开图像文件  ;
  ENVI_OPEN_FILE, "D:\test.tif", r_fid=fid
  IF (fid EQ -1) THEN BEGIN
    ENVI_BATCH_EXIT
    RETURN
  ENDIF
  ;
  ENVI_FILE_QUERY, fid, dims=dims,nb = nb
  ;对第一个波段进行计算
  pos = [0]
  ;将灰度值为1的转换为vector
  values = [1]
  ;
  l_name = 'zeroValue'
  ;evffile = FILE_DIRNAME(file)+'\img2vec.evf'
  evffile = 'D:\img2vec.evf'
  ;
  ; 栅格转换为矢量
  ;
  ENVI_DOIT, 'rtv_doit', $
    fid=fid, pos=pos, dims=dims, $
    IN_MEMORY = LINDGEN(N_ELEMENTS(values)), $
    values=values, l_name=l_name, $
    out_names=evffile
  ;evf转换为shp文件
  shapefile = 'd:\img2vec.shp'
  EVF_ID = ENVI_EVF_OPEN(evffile)
  ENVI_EVF_TO_SHAPEFILE, EVF_ID, shapefile

  ENVI_EVF_CLOSE, EVF_ID
  ; 退出ENVI
  ENVI_BATCH_EXIT
  result = FILE_TEST(shapefile, /DIRECTORY) 
  if result eq 1 then begin
    dialog=Dialog_message("栅格矢量化成功！")
  endif
END


;TODO 调用envi矢量编辑功能
PRO coastline::edit_shape
  COMPILE_OPT idl2
  ;ENVI调用初始化
  ENVI,/restore_base_save_files
  ENVI_BATCH_INIT
  ;打开EVF文件
  ;  evf_id = ENVI_EVF_OPEN("D:\img2vec.evf")
  ;  l_name = 'zeroValue'
  ;  envi_evf_info,evf_id,DATA_TYPE=integer,LAYER_NAME=l_name
  e = ENVI()
  ; Create an ENVIVector from the shapefile data
  vector1 = e.OpenVector("D:\example\pre_line.shp")
  vector = e.OpenVector("D:\example\img2vec.shp")
  raster = e.OpenRaster("D:\example\cc.jpg")
  view = e.GetView()
  layer = view.CreateLayer(vector)
  layer1 = view.CreateLayer(vector1)
  layer2 = view.CreateLayer(raster)
END
;矢量相加减
PRO coastline::vector_calculate

END
;属性赋值
PRO coastline::vector_property
  
END

;参数设置
PRO coastline::GetProperty, initFlag = initFlag
  initFlag= self.INITFLAG
END
;参数设置
PRO coastline::SetProperty, mouseType = mouseType
  self.MOUSETYPE= mouseType
END

;TODO 鼠标滚轮时的事件
PRO coastline::WheelEvents,wType,xPos,yPos
  COMPILE_OPT idl2

  self.OWINDOW.GETPROPERTY, dimensions = winDims,graphics_tree = oView
  oView.GETPROPERTY, viewPlane_Rect = viewRect

  IF wType GT 0 THEN rate = 0.8 ELSE rate = 1.125


  oriDis =[xPos,yPos]*viewRect[2:3]/winDims
  viewRect[0:1]+=(1-rate)*oriDis
  viewRect[2:3]= viewRect[2:3]*rate
  ;
  oView.SETPROPERTY, viewPlane_Rect = viewRect
  self.OWINDOW.Draw
END


;鼠标点击时的事件
PRO coastline::MousePress,xpos,ypos
  COMPILE_OPT idl2
  self.MOUSELOC[0:1] = [xPos,yPos]
  CASE self.MOUSETYPE OF
    ;放大
    2: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC[0:1],self.MOUSELOC[0:1])
      self.ORECT.SETPROPERTY, hide =0,data = data
    END
    ;缩小
    3: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC[0:1],self.MOUSELOC[0:1])
      ;void = dialog_Message(string(data),/infor)
      self.ORECT.SETPROPERTY, hide =0,data = data
    END
    ;画折线
    4: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      self.ORECT.SETPROPERTY, data = data
      self.OWINDOW.Draw
    END
    ELSE:
  ENDCASE
END


;鼠标弹起时的操作
PRO coastline::MouseRelease,xpos,ypos
  COMPILE_OPT idl2

  self.ORECT.SETPROPERTY, hide =1
  curLoc = [xPos,yPos]
  self.OWINDOW.GETPROPERTY, dimensions = winDims,graphics_tree = oView
  oView.GETPROPERTY, viewPlane_Rect = viewRect

  CASE self.MOUSETYPE OF
    ;放大
    2: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      maxV = MAX(data[0,*],min= minV)
      xRange = [minV,maxV]

      maxV = MAX(data[1,*],min= minV)
      yRange = [minV,maxV]
      ;
      viewRate= viewRect[3]/viewRect[2]
      rectRate = (yRange[1]-yRange[0])/(xRange[1]-xRange[0])
      ;
      IF viewRate GT rectRate THEN BEGIN
        width = xRange[1]-xRange[0]
        height = (xRange[1]-xRange[0])*winDims[1]/winDims[0]
        viewStartLoc= [TOTAL(xRange),TOTAL(yRange)]/2-[width,height]/2
      ENDIF ELSE BEGIN
        width = (yRange[1]-yRange[0])*winDims[0]/winDims[1]
        height = yRange[1]-yRange[0]
        viewStartLoc= [TOTAL(xRange),TOTAL(yRange)]/2-[width,height]/2
      ENDELSE
      newVP =[viewStartLoc,width,height ]
      oView.SETPROPERTY, viewPlane_Rect = newVP
    END
    ;缩小
    3: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      maxV = MAX(data[0,*],min= minV)
      xRange = [minV,maxV]

      maxV = MAX(data[1,*],min= minV)
      yRange = [minV,maxV]
      ;
      viewRate= viewRect[3]/viewRect[2]
      rectRate = (yRange[1]-yRange[0])/(xRange[1]-xRange[0])
      ;
      IF viewRate GT rectRate THEN BEGIN
        viewRect[2:3] = viewRect[2:3]*(viewRect[2])/(2*(xRange[1]-xRange[0]))
        ViewRect[0:1]= [TOTAL(xRange),TOTAL(yRange)]/2 - viewRect[2:3]/2

      ENDIF ELSE BEGIN
        viewRect[2:3] = viewRect[2:3]*(viewRect[3])/(2*(yRange[1]-yRange[0]))
        ViewRect[0:1]= [TOTAL(xRange),TOTAL(yRange)]/2 - viewRect[2:3]/2

      ENDELSE

      oView.SETPROPERTY, viewPlane_Rect = ViewRect

    END
    ;画折线
    4: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      self.ORECT.SETPROPERTY, data = data
      self.OWINDOW.Draw
    END
    ELSE:
  ENDCASE

  self.OWINDOW.Draw
END


;鼠标双击时的操作
PRO coastline::DbClick,drawId,button,xpos,ypos
  self.ORIGINALSHOW
END

PRO coastline::MouseMotion,xpos,ypos
  ;
  curLoc = [xPos,yPos]
  ;
  self.OWINDOW.GETPROPERTY, dimensions = winDims,graphics_tree = oView
  oView.GETPROPERTY, viewPlane_Rect = viewRect

  CASE self.MOUSETYPE OF
    ;平移
    1: BEGIN
      ;屏幕偏移量
      offset = curLoc- self.MOUSELOC
      ;对应偏移量
      viewRect[0:1]-=offset*viewRect[2:3]/WinDims
      oView.SETPROPERTY, viewPlane_Rect = viewRect
      self.OWINDOW.Draw
      ;
      self.MOUSELOC = curLoc
      self.OWINDOW.SETCURRENTCURSOR, 'Move'
    END
    ;放大
    2: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      self.ORECT.SETPROPERTY, data = data
      self.OWINDOW.Draw
    END
    ;缩小
    3: BEGIN
      data = self.CALRECTPOINTS(self.MOUSELOC,curLoc)
      self.ORECT.SETPROPERTY, data = data
      self.OWINDOW.Draw
    END
    ELSE:
  ENDCASE
END
;todo初始化图像显示，注意XY方向同比例变换
PRO coastline::originalShow
  ;
  self.OWINDOW.GETPROPERTY, dimensions = windowDims,graphics_tree = oView
  imageDims = self.IMAGEDIMS
  ;
  imgRate = FLOAT(imageDims[0])/imageDims[1]
  viewRate = FLOAT(windowDims[0])/windowDims[1]
  ;
  IF imgRate GT viewRate THEN BEGIN
    viewWidth = imageDims[0]
    viewHeight = imageDims[0]/viewRate
    viewPlant_rect = [0, -(viewHeight-imageDims[1])/2,viewWidth,viewHeight]

  ENDIF ELSE BEGIN
    viewHeight = imageDims[1]
    viewwidth = imageDims[1]*ViewRate
    viewPlant_rect = [-(viewwidth-imageDims[0])/2,0,viewWidth,viewHeight]

  ENDELSE
  oView.SETPROPERTY, viewPlane_Rect = viewPlant_rect,dimensions = WindowDims
  self.OWINDOW.draw
END
;todo构建显示图像体系
PRO coastline::CreateDrawImage
  oView = OBJ_NEW('IDLgrView',color = [255,255,255])
  self.OWINDOW.SETPROPERTY, graphics_tree = oView

  queryStatus = QUERY_tiff(self.INFILE, imageInfo)
  IF queryStatus EQ 0 THEN BEGIN
    self.INITFLAG= 0
    RETURN
  ENDIF

  self.IMAGEDIMS = imageInfo.DIMENSIONS
  data =READ_TIFF(self.INFILE,GEOTIFF=GeoKeys)
  ;data =READ_TIFF(self.INFILE,GEOTIFF=GeoKeys)
  TileData = BYTSCL(data) 
  self.ORIDATA = PTR_NEW(TileData,/no_Copy)
  initialLSF=2*MAKE_ARRAY(290,280,/double,VALUE=1)
  initialLSF[0:289,0:279]=-2
  phi=initialLSF
  self.CONDATA = PTR_NEW(TRANSPOSE(ROTATE(phi,1)),/no_Copy)
  self.preCONDATA = PTR_NEW(TRANSPOSE(ROTATE(phi,1)),/no_Copy)
  ;
  IF imageInfo.CHANNELS EQ 1 THEN BEGIN
    ;
    self.RGBTYPE =0
    self.OIMAGE = OBJ_NEW('IDLgrImage',*(self.ORIDATA) )

  ENDIF ELSE BEGIN
    self.RGBTYPE =1
    self.OIMAGE = OBJ_NEW('IDLgrImage',*(self.ORIDATA) ,INTERLEAVE =0)
  ENDELSE
  ;辅助红色矩形，初始化为隐藏
  self.ORECT = OBJ_NEW('IDLgrPolygon', $
    style =1,$
    thick=1,$
    color = [230,0,0])
  ;原始海岸线
  self.OPRELINE = OBJ_NEW('IDLgrPolygon', $
    style =1,$
    thick=1,$
    color = [255,0,0])
;  ;初始海岸线选择
;  self.INITLINE = OBJ_NEW('IDLgrPolyline', $
;    style =1,$
;    thick=1,$
;    color = [0,0,255])
  ;生成海岸线
  self.oCONTOUR = OBJ_NEW('IDLgrContour',$
    C_LINESTYLE=0,$
    c_label_show=0,$
    COLOR=[0,255,0] ,$
    c_value=[0,0],/hide)
  ;原始海岸线contour
  self.oPreContour = OBJ_NEW('IDLgrContour',data=*(self.preconDATA),$
    C_LINESTYLE=0,$
    c_label_show=0,$
    COLOR=[255,0,0] ,$
    c_value=255,/PLANAR, GEOMZ=0,/hide)

  oTopModel = OBJ_NEW('IDLgrModel')
  oModel = OBJ_NEW('IDLgrModel')
  lModel = OBJ_NEW('IDLgrModel')
  oModel.ADD, [self.OIMAGE,self.ORECT,self.oCONTOUR,self.oPreContour]
  lModel.add,self.OPRELINE
  ;  lModel->Rotate,[1,0,0],180
  ;  lModel->Rotate,[0,0,1],180
  oTopModel.ADD,[oModel,lModel]
  oView.Add,oTopModel
  self.ORIGINALSHOW
  self.INITFLAG= 1
END
FUNCTION coastline::INIT,infile,drawID
  ;
  self.INFILE = infile
  ;传入的drawID
;  WIDGET_CONTROL, drawID, BAD_ID=badid,GET_VALUE = oWindow
;  ;Widget_Info(drawID,GET_ = oWindow)
;  self.OWINDOW = oWindow
    self.OWINDOW = obj_new("IDLgrWindow")
  ;调用CreateImage方法创建显示图像
  self.CREATEDRAWIMAGE

  RETURN, self.INITFLAG
END

;对象类定义
PRO coastline__define
  struct = {coastline, $
    initFlag  : 0b, $
    mouseType : 0B, $;鼠标状态，0-默认,1-平移,2-放大,3-缩小。
    mouseLoc : FLTARR(2), $ ;

    infile: '' , $
    rgbType : 0, $
    imageDims : LONARR(2), $
    oriData  : PTR_NEW(), $;图像数据
    conData  : PTR_NEW(), $;生成的海岸线数据
    preConData  : PTR_NEW(), $;原始海岸线数据
    oWindow : OBJ_NEW(), $;显示窗口
    oImage  : OBJ_NEW(), $;显示图像的对象
    oRect   : OBJ_NEW(), $;放大缩小矩形对象
    oPreLine   : OBJ_NEW(), $;原始海岸线矢量对象
    oPreContour   : OBJ_NEW(), $;原始海岸线栅格对象
    oContour   : OBJ_NEW(), $;生成海岸线对象
    INITLINE   : OBJ_NEW(), $;选取初始化海岸线的对象
    DrawID: 0L $
  }
END

