# AIB from scratch

#Install the required packages if they haven't already installed
using Pkg
#=
for p in ("Knet","Images","FileIO","ImageMagick")
    #if !in(p, keys(Pkg.installed()))
    #    Pkg.add(p)
    #end
    Pkg.installed(p) == nothing && Pkg.add(p)
end
=#

using Knet, Images, FileIO, Printf
@info("used Knet, gpu=", gpu())
if gpu() < 0; @info("did not find a gpu"); end

include(Pkg.dir("Knet","data","imagenet.jl"))  #imagenet.jl includes matconvnet function etc.

using AutoGrad

using LinearAlgebra  # if using axpy!
kextrema(arr :: KnetArray) =  extrema(convert(Array{Float32}, arr))  # debugging convenience

#Define the type functions
#cpu_type = Array{Float64}
if gpu()>-1
    dtype = KnetArray{Float32}   #use KnetArray for GPU support, if there is a connected GPU.
else
    dtype = Array{Float32}   #use CPU array if no GPU is available.
end;
@info("dtype=",dtype)

#VGG_model = "imagenet-vgg-verydeep-16"
VGG_model = "imagenet-vgg-verydeep-19"
if !@isdefined vgg
    println("loading vgg weights")
    vgg = matconvnet(VGG_model)      #Load the pre-trained model(VGG-16)
end
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

# explore the vgg datastructure
vgg_meta = vgg["meta"]
vgg_classes = vgg_meta["classes"]   #{"name","description"}, the classes to categorize
vgg_normalization = vgg_meta["normalization"]   # averageIMage, cropsize, border, imageSize, interpolation, keepAspect
vgg_inputs = vgg_meta["inputs"]   # {"name"=>"data",  "size" => [224.,224.,3.,10.]}
vgg_layers = vgg["layers"]  # array[37] of entries like: 
#=
Dict{String,Any} with 9 entries:
  "name"     => "conv1_1"
  "pad"      => [1.0 1.0 1.0 1.0]
  "precious" => false
  "weights"  => Any[Float32[0.480015 0.408547 -0.0651456; 0.310477 0.0502024 -0…
  "opts"     => Array{Any}(0,0)
  "stride"   => [1.0 1.0]
  "size"     => [3.0 3.0 3.0 64.0]
  "dilate"   => 1.0
  "type"     => "conv"
=#

# this function from KnetML style transfer example
"""
    Normalizes the input image by subtracting the mean of the network
    Inputs:
    -img: an image tensor having shape of [H,W,C]
    -model_mean: the mean of the images on which the network(VGG16) has been trained.
    
    Returns:
    -img: normalized version of the input
"""
function img_normalize!(img, model_mean)
    for i in 1:size(img, 3)
        #println("img_normalize model_mean[i]=",model_mean[i])
        img[:,:,i] = (img[:,:,i] .- model_mean[i])
    end
    return img
end;

# convert from float array to rgb 8bit image
function toimg(img, addmean=true; verbose=false)
    fimg = convert(Array{Float32},img)   # size is (h,w,c,1)
    verbose && println("toimg ",extrema(fimg))
    fimg = reshape(fimg, size(fimg)[1:end-1])   # drop the last dimension
    #println("fimg size",size(fimg))
    
    if addmean
        im_mean = Array{Float32}(averageImage ./ 255)
        im_mean = [0.5f0, 0.5f0, 0.5f0]   # simplify. The means are 0.485,0.457,0.407
        img_normalize!(fimg, -1*im_mean)
        verbose && println("toimg addmean ",extrema(fimg))
    end
    
    clamp!(fimg, 0.f0,1.f0)
    #map(clamp01nan, img)  in imagemagick
    img1 = Array{FixedPointNumbers.Normed{UInt8,8}}(fimg) # size (h,w,3)
    #println("size(img1)",size(img1))
    # convert to RGB image object. size(h,w) 
    #img = colorview(RGB, permutedims(img, [1,2,3])) 
    img2 = colorview(RGB, permutedims(img1, [3,1,2]))  # because incoming size is h,w,3
    # resulting size is (h,w) with RGB type 
    #println("...toimg ",extrema(img2))
end

# convert from rgb int image to float array
# the data range differs slightly from the result of imagenet.imgdata(), not sure why.
# Maybe the order of averageImage is backward? See the "permuteddims(macfix)" code in imgdata().
# However fromimg/toimg are a round trip up to floating point
# For now, use imagenet.imgdata() to load images from disk, and use these routines subsequently.
function fromimg(img, demean=true; verbose=false) 
    # img: size (h,w) type RGB
    img1 = channelview(img)  # size (3,h,w)
    img2 = permutedims(channelview(img1),[2,3,1])  # size (w,h,3)
    fimg1 = convert(Array{Float32},img2)	# size (h,w,3),  typeof Array{Float32,3}
    verbose && println("fromimg ",extrema(fimg1))
    #println("fimg1 size",size(fimg1))
    
    if demean
        im_mean = Array{Float32}(averageImage ./ 255)
        im_mean = [0.5f0, 0.5f0, 0.5f0]
        img_normalize!(fimg1, im_mean)
        verbose && println("toimg addmean ",extrema(fimg1))        
    end
    
    fimg = reshape(fimg1, size(fimg1,1), size(fimg1,2), size(fimg1,3), 1) # add singleton dim
    verbose && println("...fromimg ",extrema(fimg))
    convert(dtype,fimg)  # back to knetarray
end



"""
    Converts an image tensor into image object stored in CPU.
    Input:
    -img:  4-dimensional image tensor. shape is [H,W,C,1]

    Returns:
    -img2: image object (to be displayed or saved)
"""
function Xpostprocess(img1)
    img = convert(Array{Float32},img1)  # convert from KnetArray, shape is (H,W,C,1)
    img = reshape(img, (size(img)...,)[1:end-1])      #shape is [H,W,C]
    println("postprocess size(img)=",size(img))
    #Denormalize the image tensor by adding the network mean
    MODEL_MEAN = Array{Float32}(averageImage ./ 255)
    println("size(MODEL_MEAN)=",size(MODEL_MEAN))
    println("MODEL_MEAN ",MODEL_MEAN[1,1,1]," ",MODEL_MEAN[1,1,2]," ",MODEL_MEAN[1,1,3])
    println("postprocess img before normalize ",minimum(img),"..",maximum(img))
    img_normalize!(img, -1*MODEL_MEAN)
    println("postprocess img after normalize ",minimum(img),"..",maximum(img))
    #clamp the tensor so that it becomes a valid image object
    clamp!(img, 0.f0,1.f0)
    #map(clamp01nan, img)  in imagemagick or Images
    img = Array{FixedPointNumbers.Normed{UInt8,8}}(img)
    img2 = colorview(RGB, permutedims(img, [3,1,2])) #convert to RGB image object. shape is [C,H,W]
    return img2
end;

#image = randn(180,240,3,1)
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
    println("averageImage=",averageImage)
    image = imgdata("cat.jpg", averageImage) ./ 255; # Array{Float32,4} (h,w,3,1) WITH MEAN REMOVED
    image = convert(dtype, image);  # KnetArray{Float32,4}
    #imgdisp = postprocess(image) #Base.ReshapedArray{RGB{Normed{UInt8,8}},2,Base.ReinterpretArray{RGB{Normed{UInt8,8}},3,Normed{UInt8,8},Array{Normed{UInt8,8},3}},Tuple{}}
    imgdisp = toimg(image) #Base.ReshapedArray{RGB{Normed{UInt8,8}},2,Base.ReinterpretArray{RGB{Normed{UInt8,8}},3,Normed{UInt8,8},Array{Normed{UInt8,8},3}},Tuple{}}
    display(imgdisp)
#image = image .- 0.5


# this code taken from knet vgg.jl code

# This procedure makes pretrained MatConvNet VGG parameters convenient for Knet
# Also, if you want to extract features, specify the last layer you want to use
"""
    Load the parameters of the pre-trained network (VGG16)
    Inputs:
    -CNN: a pre-trained model
    -atype: the type to which the network weights will be converted.
    -last_layer: the last layer that will be possibly used used for feature representation.
    
    Returns:
    -CNN parameters i.e. (weights, operations, derivatives) for all layers until last_layer
"""
function get_params(CNN, atype; last_layer="prob")
    layers = CNN["layers"]
    weights, operations, derivatives = [], [], []

    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        push!(operations, operation)
        push!(derivatives, haskey(l, "weights") && length(l["weights"]) != 0)

        if derivatives[end]
            w = copy(l["weights"])
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
        end

        last_layer != nothing && get_layer_type(last_layer) && break
    end

    map(w -> map(wi->convert(atype,wi), w), weights), operations, derivatives
end

#=
# understand what this is doing:  operation is conv,relu,conv,relu,pool,....
# I.e. the operation type, with the particular layer nubmer stripped off
    for l in vgg["layers"]
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        println(operation)
    end
=#

params = get_params(vgg,dtype)

# get convolutional network by interpreting parameters
"""
    Compute the feature representations for an input image.
    Inputs of get_convnet:
    -(weights, operations, derivatives) : Parameters of the network which were obtained using get_params function.
    Inputs of convnet:
    -img_feat: The input iamge that is fed to the network
    
    Returns:
    -outputs: Feature representations obtained from conv and relu layers until the last layer
              i.e. (conv1_1, relu1_1, con1_2, relu1_2, con2_1, relu2_1,..., relui_j) where 
              i is the last layer and j is the last sublayer
"""

tofunc(op) = eval(Meta.parse(string(op, "x")))  # append x to op, giving a function name opx
forw(x,op) = tofunc(op)(x)   # calls the function "opx" on x
forw(x,op,w) = tofunc(op)(x,w)

# the "x" functions are these
convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]
relux(x) = relu.(x)
poolx = pool
probx(x) = x
fcx(x,w) = w[1] * mat(x) .+ w[2]

# VERBOSE override above: debugging dimension mismatch when image is not 224x224:
function convx(x,w)
    rval = conv4(w[1], x; padding=1, mode=1) .+ w[2]
    println("calling conv4 size(w[1])=",size(w[1]),  " size(x)=",size(x), " => rval size ",size(rval))
    rval
end


# closure, wrap weights/operations/derivatives
# get_convnet weights=16 Array{T,2} where T[KnetArray{Float32,4}[K32(3,3,3,64)[0.4800154⋯] K32(1,1,64,1)[0.73429835⋯]], KnetArray{Float32,4}[K32(3,3,64,64)[0.16621928⋯] K32(1,1,64,1)[-0.30912212⋯]], KnetArray{Float32,4}[K32(3,3,64,128)[-0.008902963⋯] K32(1,1,128,1)[-0.04577766⋯]], KnetArray{Float32,4}[K32(3,3,128,128)[-0.0023012366⋯] K32(1,1,128,1)[-0.06418542⋯]], KnetArray{Float32,4}[K32(3,3,128,256)[0.002486315⋯] K32(1,1,256,1)[-0.074358866⋯]], KnetArray{Float32,4}[K32(3,3,256,256)[-0.0103266295⋯] K32(1,1,256,1)[0.019675018⋯]], KnetArray{Float32,4}[K32(3,3,256,256)[0.0024918285⋯] K32(1,1,256,1)[0.035415243⋯]], KnetArray{Float32,4}[K32(3,3,256,512)[-0.012453815⋯] K32(1,1,512,1)[0.013485669⋯]], KnetArray{Float32,4}[K32(3,3,512,512)[0.007116542⋯] K32(1,1,512,1)[0.023680462⋯]], KnetArray{Float32,4}[K32(3,3,512,512)[-0.0041355654⋯] K32(1,1,512,1)[0.03757768⋯]], KnetArray{Float32,4}[K32(3,3,512,512)[-0.0006068934⋯] K32(1,1,512,1)[0.09722325⋯]], KnetArray{Float32,4}[K32(3,3,512,512)[-0.0011261065⋯] K32(1,1,512,1)[0.21327873⋯]], KnetArray{Float32,4}[K32(3,3,512,512)[0.00040788297⋯] K32(1,1,512,1)[0.18837857⋯]], KnetArray{Float32,2}[K32(4096,25088)[1.9745843e-5⋯] K32(4096,1)[-0.18869764⋯]], KnetArray{Float32,2}[K32(4096,4096)[0.0039014784⋯] K32(4096,1)[0.647107⋯]], KnetArray{Float32,2}[K32(1000,4096)[0.0033880314⋯] K32(1000,1)[-0.2123236⋯]]]
# get_convnet operations=37 Any["conv", "relu", "conv", "relu", "pool", "conv", "relu", "conv", "relu", "pool", "conv", "relu", "conv", "relu", "conv", "relu", "pool", "conv", "relu", "conv", "relu", "conv", "relu", "pool", "conv", "relu", "conv", "relu", "conv", "relu", "pool", "fc", "relu", "fc", "relu", "fc", "prob"]
# get_convnet derivatives=37 Any[true, false, true, false, false, true, false, true, false, false, true, false, true, false, true, false, false, true, false, true, false, true, false, false, true, false, true, false, true, false, false, true, false, true, false, true, false]
function get_convnet(weights, operations, derivatives, skip_fc=false)
    println("get_convnet skip_fc=",skip_fc)
    println("get_convnet weights=",length(weights)," ",weights)
    println("get_convnet operations=",length(operations)," ",operations)
    println("get_convnet derivatives=",length(derivatives)," ",derivatives)

    function convnet(xs,verbose=false)
        verbose && println("convnet xs=",xs," skip_fc=",skip_fc," verbose=",verbose)
        outputs = []
        i, j = 1, 1
        num_weights, num_operations = length(weights), length(operations)
        while i <= num_operations && j <= num_weights
            skip_fc && operations[i]=="fc" && break
            if derivatives[i]
                if verbose   println(i,"   ",operations[i]," weights[j=",j,"]=",weights[j])   end
                xs = forw(xs, operations[i], weights[j])
                j += 1
            else
                xs = forw(xs, operations[i])
                if verbose   println(i,"   ",operations[i], " ",size(xs))   end
            end

            if operations[i] == "relu"
                push!(outputs, (i,xs))
            end
            
            i += 1
        end
        convert(Array{Float32}, xs), outputs
    end
end


#params = get_params(vgg, dtype)
convnet = get_convnet(params...,true)

#=  TRY CLASSIFYING  (works) 
    params = get_params(vgg, dtype)
    convnet = get_convnet(params...,false)  # false: do not skip FC layers
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"]);
    image = imgdata("cat.jpg", averageImage);
    image = convert(dtype, image);
    @info("Classifying")
    @time y1, _feats = convnet(image)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    p1 = exp.(logp(z1))
    #display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
    description = vgg["meta"]["classes"]["description"]
    display(hcat(p1[s1[1:5]], description[s1[1:5]]))
    println()
=#


#pred,feats = convnet(img)
#= feats is a list of tuples, (index,relu-activations),
(2, K32(224,224,64,1)[216.02429⋯])
 (4, K32(224,224,64,1)[0.0⋯])      
 (7, K32(112,112,128,1)[0.0⋯])     
 (9, K32(112,112,128,1)[0.0⋯])     
 (12, K32(56,56,256,1)[83.684006⋯])
 (14, K32(56,56,256,1)[84.674164⋯])
 (16, K32(56,56,256,1)[0.0⋯])      
 (19, K32(28,28,512,1)[106.7552⋯]) 
 (21, K32(28,28,512,1)[43.431267⋯])
 (23, K32(28,28,512,1)[0.0⋯])      
 (26, K32(14,14,512,1)[0.0⋯])      
 (28, K32(14,14,512,1)[0.0⋯])      
 (30, K32(14,14,512,1)[0.0⋯])      
 (33, K32(4096,1)[0.0⋯])           
 (35, K32(4096,1)[0.0⋯]) 
=#
#loss_grad = @diff convnet(img)   needs scalar loss

# test running on different image resolution... FAILS, think it is failing at the last two FC layers,
# (33, K32(4096,1)[0.0⋯])           
# (35, K32(4096,1)[0.0⋯]) 
# 224 is 14*(2^4), i.e. divide 224 by 2 4 times, gives an exact even number resulting in 14
# similarly 304=19*(2^4),  320=20*16
#img = dtype(randn(224,224,3,1))  # works
# 400 = 25*16
_img = dtype(randn(224,224,3,1))    # dummy image needed for printing the network
#img = dtype(randn(112,112,3,1))
prediction,features = convnet(_img,true)

#=  working version, image size (224,224,3,1)
convnet xs=K32(224,224,3,1)[0.59703636⋯]
calling conv4 size(w[1])=(3, 3, 3, 64) size(x)=(224, 224, 3, 1) => rval size (224, 224, 64, 1)
calling conv4 size(w[1])=(3, 3, 64, 64) size(x)=(224, 224, 64, 1) => rval size (224, 224, 64, 1)
calling conv4 size(w[1])=(3, 3, 64, 128) size(x)=(112, 112, 64, 1) => rval size (112, 112, 128, 1)
calling conv4 size(w[1])=(3, 3, 128, 128) size(x)=(112, 112, 128, 1) => rval size (112, 112, 128, 1)
calling conv4 size(w[1])=(3, 3, 128, 256) size(x)=(56, 56, 128, 1) => rval size (56, 56, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 256) size(x)=(56, 56, 256, 1) => rval size (56, 56, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 256) size(x)=(56, 56, 256, 1) => rval size (56, 56, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 512) size(x)=(28, 28, 256, 1) => rval size (28, 28, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(28, 28, 512, 1) => rval size (28, 28, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(28, 28, 512, 1) => rval size (28, 28, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(14, 14, 512, 1) => rval size (14, 14, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(14, 14, 512, 1) => rval size (14, 14, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(14, 14, 512, 1) => rval size (14, 14, 512, 1)
=#

#= dimensions that fail ... indeed, yes it was FC layers at end causing the problem
convnet xs=K32(320,320,3,1)[-1.3820721⋯]
calling conv4 size(w[1])=(3, 3, 3, 64) size(x)=(320, 320, 3, 1) => rval size (320, 320, 64, 1)
calling conv4 size(w[1])=(3, 3, 64, 64) size(x)=(320, 320, 64, 1) => rval size (320, 320, 64, 1)
calling conv4 size(w[1])=(3, 3, 64, 128) size(x)=(160, 160, 64, 1) => rval size (160, 160, 128, 1)
calling conv4 size(w[1])=(3, 3, 128, 128) size(x)=(160, 160, 128, 1) => rval size (160, 160, 128, 1)
calling conv4 size(w[1])=(3, 3, 128, 256) size(x)=(80, 80, 128, 1) => rval size (80, 80, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 256) size(x)=(80, 80, 256, 1) => rval size (80, 80, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 256) size(x)=(80, 80, 256, 1) => rval size (80, 80, 256, 1)
calling conv4 size(w[1])=(3, 3, 256, 512) size(x)=(40, 40, 256, 1) => rval size (40, 40, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(40, 40, 512, 1) => rval size (40, 40, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(40, 40, 512, 1) => rval size (40, 40, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(20, 20, 512, 1) => rval size (20, 20, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(20, 20, 512, 1) => rval size (20, 20, 512, 1)
calling conv4 size(w[1])=(3, 3, 512, 512) size(x)=(20, 20, 512, 1) => rval size (20, 20, 512, 1)
=#

function mkreludictionary()
    # for VGG16: dimension 29; for VGG19: dimension 35.
    reludict = -1 * ones(Int32,35)  # index from desired relu back to featuremap index
#    reludict = -1 * ones(Int32,29)  # index from desired relu back to featuremap index
    prediction, features = convnet(_img)
    println("CORRESPONDENCE of feature# and pytorch relu:")
    for ifeat = 1:length(features)
        irelu = features[ifeat][1]-1
        println(ifeat," relu ",irelu)
        reludict[irelu] = ifeat
    end
    reludict
end
ReluDict = mkreludictionary()
@assert ReluDict[11]==5
@assert ReluDict[13]==6

function loss(img)
    compatscale = 64.f0*size(img,1)*size(img,2)   # poor choice but compatible with nips17 experiment naming
    prediction,features = convnet(img)
    relu = features[6][2]
    -sum(relu .* relu) / compatscale
end

function PROBELOSS(img)
    #tmpimg = convert(Array{Float32},img)
    #println("img 50,50,: ",tmpimg[50,50,1:3,1], " 51,51,: ", tmpimg[51,51,1:3,1])

    prediction,features = convnet(img)
    #layers = [11,13]
    layers = [13]
    theloss = 0.f0
    
    _feat = features[ReluDict[layers[1]]][2]   # poor choice but compatible with nips17 experiment naming
    println("size(_feat)=",size(_feat))
    compatscale = reduce(*,size(_feat))   # should be recomputed per layer
    println("compatscale=",compatscale)
    
    for irelu in layers
        println("looping irelu=",irelu)
        ifeat = ReluDict[irelu]
        @assert ifeat > 0
        relu = features[ifeat][2]
        println("features[ifeat,1]=",features[ifeat][1]," iRelu=",irelu," -> ","ifeat=",ifeat)
        println("size(feat)=",size(relu), " typeof=",typeof(relu))
        frelu = convert(Array{Float32},relu)
        println("extrema(relu)=",extrema(frelu))
        theloss1 = sum(relu .* relu)
        theloss1 /= compatscale
        println("theloss=",theloss1)
        println("f theloss=", sum(frelu .* frelu)/compatscale)
        theloss -= theloss1
    end
    theloss
end

function loss(img)
    prediction,features = convnet(img)
    #layers = [11,13]
    layers = [13]
    theloss = 0.f0
    
    _feat = features[ReluDict[layers[1]]][2]   # poor choice but compatible with nips17 experiment naming
    compatscale = reduce(*,size(_feat))   # should be recomputed per layer
    
    for irelu in layers
        ifeat = ReluDict[irelu]
        @assert ifeat > 0
        relu = features[ifeat][2]
        theloss1 = sum(relu .* relu)
        theloss1 /= compatscale
        theloss -= theloss1
    end
    theloss
end

# redefine non-verbose version
convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]


# comparison 
# python AIBsynth.py --imgsize 400 --save_iter 100 --niter 1001  --layers 11 13  --outfile _AIB_relu_11_13_%04d.png
# python AIBsynth.py --imgsize 400 --save_iter 50 --niter 1001 --blur_iter 100 --layers 11 13  --outfile _AIB_bl_100_relu_11_13__%04d.png


gRandSeed = 101
gDisplayIter = 100
gBlur = -1
gLr = 1.f0

gLr = 0.3f0

import Random
(gRandSeed > 0) && Random.seed!(gRandSeed)

# fimg is size (h,w,3,1)
fimg = copy(image)     # image is original, fimg is evolved
fimg = dtype(0.5f0 * randn(size(image)))  #gaussian noise (mean:0, var:1)  var(1) gives extreme values out to ~4.5
fimg = dtype(0.5f0 * randn(400,400,3,1))
fimg = dtype(0.5f0 * randn(224,224,3,1)) 
#fimg = Param(fimg)
imgdisp = toimg(fimg)
#display(imgdisp)
#save("_startingimg.png",imgdisp)

# reproducable comparison
imgdisp = load("_startingimg.png")
#imgdisp = load("cat.jpg")
fimg = fromimg(imgdisp)
display(imgdisp)

for iter=1:1001
    verboseiter = (iter%10)==0
    verboseiter && println("iteration ",iter)
    
    fimg = Param(fimg)

    dloss = @diff loss(fimg)  # typeof(dloss) = AutoGrad.Tape 
    g = grad(dloss,fimg)
    #LinearAlgebra.axpy!(-gLr, g, fimg) # Linalg version does not work for knetarray
    #@. img += -0.0001 * g
    fimg .-= gLr * g
    #update!(fimg, g, SGD(lr=gLr)) # does not work. no method matching length(::SGD)
    
    verboseiter && println(" loss=",value(dloss))
    
    blurtime = (gBlur > 0) && ((iter % gBlur)==0)
    #println("gBlur=",gBlur," iter%gBlur=",(iter % gBlur)," blurtime = ",blurtime)
    iimg = toimg(fimg, verbose=blurtime)  # clamps it
    if blurtime
        println("blurring")
        iimg = imfilter(iimg, Kernel.gaussian(3));
        iimg = imadjustintensity(iimg) # expands to 0,1 range
    end
    fimg = fromimg(iimg,verbose=blurtime)
    
    #img = imgclamp(img)
    
    if (iter-1)%gDisplayIter == 0
        println("typeof(img)=",typeof(fimg)," typeof g=",typeof(g))
        println("img in ",minimum(fimg),"..",maximum(fimg), "  g in ", minimum(g),"..",maximum(g)) # todo extrema not defined for knetarray
        display(toimg(fimg,verbose=true))
        #map(clamp01nan, img)
        save(@sprintf("_output.%04d.jpg",iter),toimg(fimg))
        #display(postprocess(g))
        save(@sprintf("_grad.%04d.jpg",iter),toimg(g))
    end
end

# gradient check.  
# Generally obtains correlation of ~0.95 between single-pixel loss increment and approximation of + grad*eps
import Random
Random.seed!(101)
junkimg = dtype(randn(400,400,3,1));
junkimg = Param(junkimg)

# 1. do the Autograd numeric chec,
AutoGrad.gcheck(loss,junkimg; verbose=2, nsample=10, delta=0.1)

# 2. modify single pixels that have large derivative,
# compare the resulting loss L2 to the result from   L1 + AD*junkeps  
# where L1 is the original loss, AD is the autodiff derivative.

println("")
_dloss = @diff loss(junkimg)
_g = grad(_dloss,junkimg);

function vnorm(v)
    v .-= sum(v) / length(v)
    v ./= norm(v)
end

for jj=200:210
    vdiff = []  # record some differentials
    vval = []   # record some value differences
    
    for ii=10:390 
        thegrad = _g[jj,ii,2,1]   # =>  0.0031
        #if abs(thegrad) < 0.006  continue;  end  #vgg16
        if abs(thegrad) < 0.002  continue;  end  # vgg19
        junkimg[jj,ii,2,1]  # -0.986
        L1 = loss(junkimg)  # -249.1141f0    # get the original loss
        junkeps = 1.f0
        differential = _g[jj,ii,2,1] * junkeps
        junkimg[jj,ii,2,1] += junkeps    # change this pixel of the image
        L2 = loss(junkimg)  # -249.1113f0    # get the new loss
        valdiff = L2 - L1   # 0.0028076172f0
        println("obtained=",valdiff, " vs grad*eps = ", differential)
        append!(vval,valdiff)
        append!(vdiff,differential)
    end
    
    # if vval is empty it means the abs(thegrad) threshold is too harsh
    println("correlation = ", sum(vnorm(vdiff) .* vnorm(vval)))
end




