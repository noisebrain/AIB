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

#Define the type functions
#cpu_type = Array{Float64}
if gpu()>-1
    dtype = KnetArray{Float32}   #use KnetArray for GPU support, if there is a connected GPU.
else
    dtype = Array{Float32}   #use CPU array if no GPU is available.
end;
@info("dtype=",dtype)

VGG_model = "imagenet-vgg-verydeep-16"
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

# this code from KnetML style transfer example
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
        println("img_normalize model_mean[i]=",model_mean[i])
        img[:,:,i] = (img[:,:,i] .- model_mean[i])
    end
    return img
end;

# clamp!(::KnetArray{Float32,3}

"""
    Converts an image tensor into image object stored in CPU.
    Input:
    -img:  4-dimensional image tensor. shape is [H,W,C,1]

    Returns:
    -img2: image object (to be displayed or saved)
"""
function postprocess(img1)
    img = convert(Array{Float32},img1)  # convert from KnetArray
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
    #map(clamp01nan, img)  in imagemagick
    img = Array{FixedPointNumbers.Normed{UInt8,8}}(img)
    img2 = colorview(RGB, permutedims(img, [3,1,2])) #convert to RGB image object. shape is [C,H,W]
    return img2
end;

#image = randn(180,240,3,1)
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"]);
    image = imgdata("cat.jpg", averageImage) ./ 255; # Array{Float32,4} (h,w,3,1)
    image = convert(dtype, image);  # KnetArray{Float32,4}
    imgdisp = postprocess(image) #Base.ReshapedArray{RGB{Normed{UInt8,8}},2,Base.ReinterpretArray{RGB{Normed{UInt8,8}},3,Normed{UInt8,8},Array{Normed{UInt8,8},3}},Tuple{}}
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
        println("convnet xs=",xs," skip_fc=",skip_fc," verbose=",verbose)
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
                if verbose   println(i,"   ",operations[i])   end
                xs = forw(xs, operations[i])
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
img = dtype(randn(320,320,3,1))  # seems like 224 is the only size that works
#img = dtype(randn(112,112,3,1))
prediction,features = convnet(img,true)

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

#= dimensions that fail
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
    reludict = -1 * ones(Int32,29)  # index from desired relu back to featuremap index
    prediction, features = convnet(img)
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
    prediction,features = convnet(img)
    relu = features[6][2]
    -sum(relu .* relu)
end

function loss(img)
    prediction,features = convnet(img)
    layers = [11,13]
    theloss = 0.f0
    for irelu in layers
        ifeat = ReluDict[irelu]
        @assert ifeat > 0
        relu = features[ifeat][2]
        theloss -= sum(relu .* relu)
    end
    theloss
end

function imgclamp(img)
    img1 = convert(Array{Float32},img)
    img1 = clamp.(img1, -0.5f0, 0.5f0)
    img = convert(KnetArray{Float32},img1)
    img = Param(img)
    return img
end

# todo some faster / blas / parallelizable way?
#  20.484436 seconds (14.47 M allocations: 437.874 MiB, 0.34% gc time)
function imgclamp!(img)
    println("before clamp, img ",minimum(img),"...",maximum(img))
    dims = size(img)
    @assert dims[4]==1
    len = reduce(*,dims)
    #rintln("len = ",len)
    @inbounds for i=1:len
        v = img[i]
        if v > 0.5f0   
            img[i] = 0.5f0 
        elseif v < -0.5f0
            img[i] = -0.5f0
        end
    end
    println("after clamp, img ",minimum(img),"...",maximum(img))
end

# redefine non-verbose version
convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]

img = copy(image)     # image is original, img is evolved
img = dtype(randn(size(image)))  #gaussian noise (mean:0, var:1)
img = dtype(randn(600,600,3,1))
img = Param(img)
imgdisp = postprocess(img)
display(imgdisp)

for iter=1:500
    println("iteration ",iter)
    dloss = @diff loss(img)
    g = grad(dloss,img)
    #Base.axpy!(-0.01, g, img)
    #@. img += -0.0001 * g
    img .+= -0.000005f0 * g
    #img = clamp.(img, -0.5f0, 0.5f0)
    img = imgclamp(img)
    
    if (iter-1)%5 == 0
        println("typeof(img)=",typeof(img)," typeof g=",typeof(g))
        println("img in ",minimum(img),"..",maximum(img), "  g in ", minimum(g),"..",maximum(g))
        display(postprocess(img))
        #map(clamp01nan, img)
        save(@sprintf("_output.%04d.jpg",iter),postprocess(img))
        #display(postprocess(g))
        save(@sprintf("_grad.%04d.jpg",iter),postprocess(g))
    end
end




