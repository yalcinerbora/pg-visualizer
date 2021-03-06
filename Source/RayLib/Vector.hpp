template <int N, class T>
template<class C, typename>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::Vector(C data)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data);
    }
}

template <int N, class T>
template<class C, typename>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::Vector(const C* data)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data[i]);
    }
}

template <int N, class T>
template <class... Args, typename>
__device__ __host__ HYBRID_INLINE
constexpr Vector<N, T>::Vector(const Args... dataList)
    : vector{static_cast<T>(dataList) ...}
{
    static_assert(sizeof...(dataList) == N, "Vector constructor should have exact "
                  "same template count "
                  "as arguments");
}

template <int N, class T>
template <class... Args, typename>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::Vector(const Vector<N - sizeof...(Args), T>& v,
                     const Args... dataList)
{
    constexpr int vectorSize = N - sizeof...(dataList);
    static_assert(sizeof...(dataList) + vectorSize == N, "Total type count of the partial vector"
                  "constructor should exactly match vector size");

    UNROLL_LOOP
    for(int i = 0; i < vectorSize; i++)
    {
        vector[i] = v[i];
    }
    const T arr[] = {static_cast<T>(dataList)...};
    UNROLL_LOOP
    for(int i = vectorSize; i < N; i++)
    {
        vector[i] = arr[i - vectorSize];
    }
}

template <int N, class T>
template <int M, typename>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::Vector(const Vector<M, T>& other)
{
    static_assert(M > N, "enable_if sanity check.");
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = other[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::operator T* ()
{
    return vector;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::operator const T* () const
{
    return vector;
}

template <int N, class T>
template<int M, class C, typename>
__device__ __host__ HYBRID_INLINE
Vector<N, T>::operator Vector<M, C>() const
{
    Vector<M, C> result;
    UNROLL_LOOP
    for(int i = 0; i < M; i++)
    {
        result[i] = static_cast<C>(vector[i]);
    }
    return result;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T& Vector<N, T>::operator[](int i)
{
    return vector[i];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
constexpr const T& Vector<N, T>::operator[](int i) const
{
    return vector[i];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator+=(const Vector& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] += right[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator-=(const Vector& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] -= right[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator*=(const Vector& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] *= right[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator*=(T right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] *= right;
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator/=(const Vector& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] /= right[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Vector<N, T>::operator/=(T right)
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] /= right;
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator+(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] + right[i];
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator+(T r) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] + r;
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator-(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] - right[i];
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator-(T r) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] - r;
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Vector<N, T>> Vector<N, T>::operator-() const
{
    Vector<N, T> v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = -vector[i];
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator*(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right[i];
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator*(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right;
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator/(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right[i];
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::operator/(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right;
    }
    return v;
}

template <int N, class T>
template<class Q>
__device__ __host__ HYBRID_INLINE
IntegralEnable<Q, Vector<N, T>> Vector<N, T>::operator%(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right[i];
    }
    return v;
}

template <int N, class T>
template<class Q>
__device__ __host__ HYBRID_INLINE
IntegralEnable<Q, Vector<N, T>> Vector<N, T>::operator%(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right;
    }
    return v;
}

template <int N, class T>
template<class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::operator%(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right[i]);
    }
    return v;
}

template <int N, class T>
template<class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::operator%(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator==(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        b &= vector[i] == right[i];
    }
    return b;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator!=(const Vector& right) const
{
    return !(*this == right);
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator<(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        b &= vector[i] < right[i];
    }
    return b;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator<=(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        b &= vector[i] <= right[i];
    }
    return b;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator>(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        b &= vector[i] > right[i];
    }
    return b;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::operator>=(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        b &= vector[i] >= right[i];
    }
    return b;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T Vector<N, T>::Dot(const Vector& right) const
{
    T data = 0;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        data += (vector[i] * right[i]);
    }
    return data;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T Vector<N, T>::Sum() const
{
    T result = 0;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        result += vector[i];
    }
    return result;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T Vector<N, T>::Multiply() const
{
    T result = 1;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        result *= vector[i];
    }
    return result;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
int Vector<N, T>::Max() const
{
    int result = 0;
    T max = vector[0];
    UNROLL_LOOP
    for(int i = 1; i < N; i++)
    {
        if(vector[i] > max)
        {
            max = vector[i];
            result = i;
        }
    }
    return result;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
int Vector<N, T>::Min() const
{
    int result = 0;
    T min = vector[0];
    UNROLL_LOOP
    for(int i = 1; i < N; i++)
    {
        if(vector[i] < min)
        {
            min = vector[i];
            result = i;
        }
    }
    return result;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, T> Vector<N, T>::Length() const
{
    return sqrt(LengthSqr());
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T Vector<N, T>::LengthSqr() const
{
    return Dot(*this);
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::Normalize() const
{
    T lengthInv = static_cast<T>(1) / Length();

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = vector[i] * lengthInv;
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>&> Vector<N, T>::NormalizeSelf()
{
    T lengthInv = static_cast<T>(1) / Length();
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] *= lengthInv;
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Clamp(const Vector& minVal, const Vector& maxVal) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = min(max(minVal[i], vector[i]), maxVal[i]);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Clamp(T minVal, T maxVal) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = min(max(minVal, vector[i]), maxVal);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T>& Vector<N, T>::ClampSelf(const Vector& minVal, const Vector& maxVal)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = min(max(minVal[i], vector[i]), maxVal[i]);
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T>& Vector<N, T>::ClampSelf(T minVal, T maxVal)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = min(max(minVal, vector[i]), maxVal);
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Vector<N, T>::HasNaN() const
{
    bool hasNan = false;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        hasNan |= ((vector[i] != vector[i]) ||
                   (vector[i] == INFINITY) ||
                   (vector[i] == -INFINITY));
    }
    return hasNan;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Vector<N, T>> Vector<N, T>::Abs() const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = abs(vector[i]);
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Vector<N, T>&> Vector<N, T>::AbsSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = abs(vector[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::Round() const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = round(vector[i]);
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>&> Vector<N, T>::RoundSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = round(vector[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::Floor() const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = floor(vector[i]);
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>&> Vector<N, T>::FloorSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = floor(vector[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::Ceil() const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = ceil(vector[i]);
    }
    return v;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>&> Vector<N, T>::CeilSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        vector[i] = ceil(vector[i]);
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Min(const Vector& v0, const Vector& v1)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = min(v0[i], v1[i]);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Min(const Vector& v0, T v1)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = min(v0[i], v1);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Max(const Vector& v0, const Vector& v1)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = max(v0[i], v1[i]);
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Vector<N, T> Vector<N, T>::Max(const Vector& v0, T v1)
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = max(v0[i], v1);
    }
    return v;
}

template <int N, class T>
template<class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Vector<N, T>> Vector<N, T>::Lerp(const Vector& v0, const Vector& v1, T t)
{
    assert(t >= 0 && t <= 1);
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = (1 - t) * v0[i] + t * v1[i];
    }
    return v;
}

template<int N, class T>
__device__ __host__
Vector<N, T> operator*(T left, const Vector<N, T>& vec)
{
    return vec * left;
}

// Cross product (only for 3d vectors)
template<class T>
__device__ __host__
Vector<3, T> Cross(const Vector<3, T>& v0, const Vector<3, T>& v1)
{
    Vector<3, T> result(v0[1] * v1[2] - v0[2] * v1[1],
                        v0[2] * v1[0] - v0[0] * v1[2],
                        v0[0] * v1[1] - v0[1] * v1[0]);
    return result;
}

template <class T>
__device__ __host__
Vector<3, T> OrthogonalVector(const Vector<3, T>& v)
{
    // PBRT Book
    // https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors#CoordinateSystem

    if(abs(v[0]) > abs(v[1]))
        return Vector<3, T>(-v[2], 0, v[0]) / sqrt(v[0] * v[0] + v[2] * v[2]);
    else
        return Vector<3, T>(0, v[2], -v[1]) / sqrt(v[1] * v[1] + v[2] * v[2]);
}